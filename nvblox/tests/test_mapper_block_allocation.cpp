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
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "nvblox/core/types.h"
#include "nvblox/geometry/bounding_spheres.h"
#include "nvblox/io/image_io.h"
#include "nvblox/io/mesh_io.h"
#include "nvblox/io/pointcloud_io.h"
#include "nvblox/mapper/mapper.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/tests/utils.h"

using namespace nvblox;

//
// ==================== PARAMETRIZATION STRUCT ====================
//

template <typename LayerTypeT, EsdfMode esdf_mode, bool has_freespace_layer>
struct ParametrizationStruct {
  using LayerType = LayerTypeT;
  static EsdfMode getEsdfMode() { return esdf_mode; }
  static bool hasFreespaceLayer() { return has_freespace_layer; }
};

//
// ==================== TEST CLASS DEFINTION ====================
//

template <typename ParametrizationStruct>
struct MapperTestBlocksInLayers : public testing::Test {
 protected:
  using LayerType = typename ParametrizationStruct::LayerType;

  /// @brief Initialize the scene.
  void SetUp() override {
    // Setup a simple scene with a sphere inside a box.
    scene_.aabb() = AxisAlignedBoundingBox(Vector3f(-5.0f, -5.0f, -5.0f),
                                           Vector3f(5.0f, 5.0f, 5.0f));
    scene_.addGroundLevel(-4.0f);
    scene_.addCeiling(4.0f);
    scene_.addPrimitive(
        std::make_unique<primitives::Sphere>(Vector3f(0.0, 0.0, 0.0), 2.0));
    scene_.addPlaneBoundaries(-4.0f, 4.0f, -4.0f, 4.0f);

    // Setup the slice bounds to include the sphere.
    mapper_.esdf_integrator().esdf_slice_min_height(-3.0f);
    mapper_.esdf_integrator().esdf_slice_max_height(3.0f);
    mapper_.esdf_integrator().esdf_slice_height(0.0f);
  }

  /// @brief Load the projective layer from the scene and then update all the
  /// other layers.
  /// @return The number of allocated blocks in the projective layer.
  int loadLayersFromScene() {
    // Load the occupancy/tsdf layer from the sample scene.
    LayerType layer_host(kVoxelSizeM, MemoryType::kHost);
    scene_.generateLayerFromScene(1.0, &layer_host);
    mapper_.layers().getPtr<LayerType>()->copyFrom(layer_host);

    // Now update all the other layers.
    updateLayers(UpdateFullLayer::kYes);

    return getNumAllocatedProjectiveBlocks();
  }

  /// @brief Generate a depth image from the scene and integrate it into the
  /// projective layer. Then update all the other layers.
  /// @return The number of allocated blocks in the projective layer.
  int loadLayersFromFrame() {
    // Get a camera position to observe the sample scene.
    Camera camera(300, 300, 320, 240, 640, 480);
    Eigen::Quaternionf rotation_base(0.5, 0.5, 0.5, 0.5);
    Eigen::Vector3f translation(0.0, 0.0, 0.0);
    Transform T_S_C = Transform::Identity();
    T_S_C.prerotate(rotation_base);
    T_S_C.pretranslate(translation);

    // Get a depth image from the sample scene.
    DepthImage depth_frame(camera.height(), camera.width(),
                           MemoryType::kUnified);
    constexpr float kMaxDistM = 10.0;
    scene_.generateDepthImageFromScene(camera, T_S_C, kMaxDistM, &depth_frame);

    // Integrate the depth image and update all other layers.
    mapper_.integrateDepth(depth_frame, T_S_C, camera);
    updateLayers();

    return getNumAllocatedProjectiveBlocks();
  }

  /// @brief Test that all layers have the right blocks allocated.
  void testBlocksInLayers() {
    EXPECT_GT(mapper_.layers().get<LayerType>().numBlocks(), 0);

    // Check color layer.
    // It should always have the same blocks as the projective layer
    // (tsdf/occupancy).
    EXPECT_EQ(mapper_.layers().get<LayerType>().numBlocks(),
              mapper_.color_layer().numBlocks());
    for (const Index3D& projective_idx :
         mapper_.layers().get<LayerType>().getAllBlockIndices()) {
      EXPECT_TRUE(mapper_.color_layer().isBlockAllocated(projective_idx));
    }

    // Check freespace layer (if existent).
    // It should always have the same blocks as the projective layer
    // (tsdf/occupancy).
    if (ParametrizationStruct::hasFreespaceLayer()) {
      EXPECT_EQ(mapper_.layers().get<LayerType>().numBlocks(),
                mapper_.freespace_layer().numBlocks());
      for (const Index3D& projective_idx :
           mapper_.layers().get<LayerType>().getAllBlockIndices()) {
        EXPECT_TRUE(mapper_.freespace_layer().isBlockAllocated(projective_idx));
      }
    }

    // Check Esdf layer.
    // If EsdfMode is 3D, it should always have the same blocks as the
    // projective layer (tsdf/occupancy).
    if (ParametrizationStruct::getEsdfMode() == EsdfMode::k3D) {
      EXPECT_EQ(mapper_.layers().get<LayerType>().numBlocks(),
                mapper_.esdf_layer().numBlocks());
      for (const Index3D& projective_idx :
           mapper_.layers().get<LayerType>().getAllBlockIndices()) {
        EXPECT_TRUE(mapper_.esdf_layer().isBlockAllocated(projective_idx));
      }
    }

    // In case of 2d Esdf mode the test has to be more involved.
    if (ParametrizationStruct::getEsdfMode() == EsdfMode::k2D) {
      EXPECT_GE(mapper_.layers().get<LayerType>().numBlocks(),
                mapper_.esdf_layer().numBlocks());
      EXPECT_GT(mapper_.esdf_layer().numBlocks(), 0);

      // Get the slice bound indices.
      const float block_size = mapper_.esdf_layer().block_size();
      const int output_slice_block_index_z =
          getBlockIndexFromPositionInLayer(
              block_size,
              Vector3f(0.0f, 0.0f,
                       mapper_.esdf_integrator().esdf_slice_height()))
              .z();
      const int min_input_block_index_z =
          getBlockIndexFromPositionInLayer(
              block_size,
              Vector3f(0.0f, 0.0f,
                       mapper_.esdf_integrator().esdf_slice_min_height()))
              .z();
      const int max_input_block_index_z =
          getBlockIndexFromPositionInLayer(
              block_size,
              Vector3f(0.0f, 0.0f,
                       mapper_.esdf_integrator().esdf_slice_max_height()))
              .z();

      // Test that for every 3d projective block inside the slice bounds there
      // is a 2d esdf block.
      for (const Index3D& projective_idx :
           mapper_.layers().get<LayerType>().getAllBlockIndices()) {
        if (projective_idx.z() >= min_input_block_index_z &&
            projective_idx.z() <= max_input_block_index_z) {
          EXPECT_TRUE(mapper_.esdf_layer().isBlockAllocated(
              Index3D(projective_idx.x(), projective_idx.y(),
                      output_slice_block_index_z)));
        }
      }
      // Test that every 2d esdf block has at least one 3d projective block.
      for (const Index3D& esdf_idx :
           mapper_.esdf_layer().getAllBlockIndices()) {
        bool has_corresponding_projective_block = false;
        for (const Index3D& projective_idx :
             mapper_.layers().get<LayerType>().getAllBlockIndices()) {
          if (esdf_idx.x() == projective_idx.x() &&
              esdf_idx.y() == projective_idx.y()) {
            has_corresponding_projective_block = true;
            break;
          }
        }
        EXPECT_TRUE(has_corresponding_projective_block);
      }
    }

    // Check Mesh layer.
    // Not for every occupancy/tsdf block we have a mesh block.
    // NOTE: We only produce a mesh when the layer type is tsdf.
    if (std::is_same<LayerType, TsdfLayer>::value) {
      // Checks for color mesh layer.
      EXPECT_GE(mapper_.layers().get<LayerType>().numBlocks(),
                mapper_.color_mesh_layer().numBlocks());
      EXPECT_GT(mapper_.color_mesh_layer().numBlocks(), 0);
      for (const Index3D& mesh_idx :
           mapper_.color_mesh_layer().getAllBlockIndices()) {
        // For every mesh block there should be a block in the
        // projective layer.
        EXPECT_TRUE(
            mapper_.layers().get<LayerType>().isBlockAllocated(mesh_idx));
      }

      // Checks for the feature mesh layer.
      EXPECT_GE(mapper_.layers().get<LayerType>().numBlocks(),
                mapper_.feature_mesh_layer().numBlocks());
      EXPECT_GT(mapper_.feature_mesh_layer().numBlocks(), 0);
      for (const Index3D& mesh_idx :
           mapper_.feature_mesh_layer().getAllBlockIndices()) {
        // For every mesh block there should be a block in the
        // projective layer.
        EXPECT_TRUE(
            mapper_.layers().get<LayerType>().isBlockAllocated(mesh_idx));
      }
    }
  }

  /// @return The number of allocated blocks in the projective layer.
  int getNumAllocatedProjectiveBlocks() {
    return mapper_.layers().get<LayerType>().numBlocks();
  }

  /// @brief Get the projective layer type from the class parametrization.
  /// @return The projective layer type
  ProjectiveLayerType getProjectiveLayerType() {
    if (std::is_same<LayerType, TsdfLayer>::value) {
      if (ParametrizationStruct::hasFreespaceLayer()) {
        return ProjectiveLayerType::kTsdfWithFreespace;
      } else {
        return ProjectiveLayerType::kTsdf;
      }
    } else {
      return ProjectiveLayerType::kOccupancy;
    }
  }

  /// @brief Decay the projective layer.
  void decayLayers() {
    if (std::is_same<LayerType, TsdfLayer>::value) {
      mapper_.decayTsdfAllVoxels();
    } else {
      mapper_.decayOccupancyAllVoxels();
    }
  }

  /// @brief Update color, mesh, freespace and esdf layers.
  /// @param update_full_layer Whether to update all blocks of the layers.
  void updateLayers(UpdateFullLayer update_full_layer = UpdateFullLayer::kNo) {
    // Allocate color blocks as we can't get it from the scene.
    for (const Index3D& idx :
         mapper_.layers().get<LayerType>().getAllBlockIndices()) {
      mapper_.color_layer().allocateBlockAtIndex(idx);
      mapper_.feature_layer().allocateBlockAtIndex(idx);
    }
    mapper_.updateColorMesh(update_full_layer);
    mapper_.updateFeatureMesh(update_full_layer);
    if (ParametrizationStruct::hasFreespaceLayer()) {
      mapper_.updateFreespace(Time(0), update_full_layer);
    }
    if (ParametrizationStruct::getEsdfMode() == EsdfMode::k3D) {
      mapper_.updateEsdf(update_full_layer);
    } else {
      mapper_.updateEsdfSlice(update_full_layer);
    }
  }

  static constexpr float kVoxelSizeM = 0.1F;
  primitives::Scene scene_;
  Mapper mapper_{kVoxelSizeM, MemoryType::kDevice, getProjectiveLayerType()};
};

//
// ==================== TEST PARAMETRIZATION ====================
//

// We are testing all combinations of esdf mode and projective layer type.
// NOTE: Freespace is only valid in combination with tsdf.
using MapperTestBlocksInLayersTypes = ::testing::Types<
    ParametrizationStruct<TsdfLayer, EsdfMode::k3D, false>,
    ParametrizationStruct<OccupancyLayer, EsdfMode::k3D, false>,
    ParametrizationStruct<TsdfLayer, EsdfMode::k2D, false>,
    ParametrizationStruct<OccupancyLayer, EsdfMode::k2D, false>,
    ParametrizationStruct<TsdfLayer, EsdfMode::k3D, true>,
    ParametrizationStruct<TsdfLayer, EsdfMode::k2D, true>>;

TYPED_TEST_SUITE(MapperTestBlocksInLayers, MapperTestBlocksInLayersTypes);

//
// ==================== TEST DEFINITIONS ====================
//

TYPED_TEST(MapperTestBlocksInLayers, LoadFromScene) {
  // Load a layer (tsdf/occupancy) from a scene and update color/esdf/mesh.
  this->loadLayersFromScene();
  // Check that all layers have the blocks they should.
  this->testBlocksInLayers();
}

TYPED_TEST(MapperTestBlocksInLayers, LoadFromFrame) {
  // Update a layer (tsdf/occupancy) from a depth image and update
  // color/esdf/mesh.
  this->loadLayersFromFrame();
  // Check that all layers have the blocks they should.
  this->testBlocksInLayers();
}

TYPED_TEST(MapperTestBlocksInLayers, ClearOutsideRadius) {
  // Load a layer (tsdf/occupancy) from a scene and update color/esdf/mesh.
  const int num_blocks = this->loadLayersFromScene();

  // Deleting blocks outside a radius.
  const Vector3f sphere_center(0.0f, 0.0f, 0.0f);
  const float sphere_radius = 1.0f;
  this->mapper_.clearOutsideRadius(sphere_center, sphere_radius);

  // Again, all layers should have the same block count.
  this->testBlocksInLayers();
  // After clearing some blocks, we should have less than at the beginning.
  EXPECT_LT(this->getNumAllocatedProjectiveBlocks(), num_blocks);
}

TYPED_TEST(MapperTestBlocksInLayers, DecayProjectiveLayer) {
  // Load a layer (tsdf/occupancy) from a scene and update color/esdf/mesh.
  const int num_blocks = this->loadLayersFromScene();

  // Next we setup occupancy/tsdf decay to decay some of the blocks (but not
  // all).

  // Tsdf decay:
  // 1) We set the decay factor to deallocate blocks after one decay step.
  // 2) We integrate a frame to increase the weights of some blocks
  // 3) When the decay is applied, only the blocks not visible in the
  // integrated
  //    frame get deallocated.
  this->mapper_.tsdf_decay_integrator().decay_factor(
      kTsdfDecayedWeightThresholdDesc.default_value);
  this->mapper_.tsdf_integrator().max_weight(
      kTsdfDecayedWeightThresholdDesc.default_value +
      WeightingFunction::kDefaultConstantWeight);
  this->mapper_.tsdf_integrator().weighting_function_type(
      WeightingFunctionType::kConstantWeight);

  // Occupancy decay:
  // 1) We increase the decay probability for the blocks in free region to
  //    deallocate after one decay step.
  // 2) When the decay is applied, blocks in the free region are deallocated
  //    and while blocks in occupied region are not.
  constexpr float kFreeRegionDecayProb = 0.999f;
  this->mapper_.occupancy_decay_integrator().free_region_decay_probability(
      kFreeRegionDecayProb);

  // Integrate a frame to increase the weight of some voxels.
  this->loadLayersFromFrame();
  // We haven't decayed yet, so we expect to have the same number of blocks as
  // in the beginning.
  this->testBlocksInLayers();
  EXPECT_EQ(this->getNumAllocatedProjectiveBlocks(), num_blocks);

  // Do the decay.
  this->decayLayers();
  // Again, all layers should have the same block count (also after another
  // update).
  this->testBlocksInLayers();
  this->updateLayers();
  this->testBlocksInLayers();

  // After decaying some blocks, we should have less than at the beginning.
  EXPECT_LT(this->getNumAllocatedProjectiveBlocks(), num_blocks);
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
