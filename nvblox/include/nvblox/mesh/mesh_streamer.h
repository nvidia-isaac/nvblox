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
#pragma once

#include <memory>
#include <optional>

#include "nvblox/core/hash.h"
#include "nvblox/core/parameter_tree.h"
#include "nvblox/core/types.h"
#include "nvblox/map/common_names.h"
#include "nvblox/mesh/mesh_block.h"
#include "nvblox/serialization/mesh_serializer.hpp"

namespace nvblox {

/// @brief Helps manage bandwidth used during mesh streaming.
///
/// Note that this class is an abstract base class. Child classes have to
/// implement the compute priority function, which determines *which* blocks are
/// streamed when the number of potential blocks exceeds the requested limit.
class MeshStreamerBase {
 public:
  /// @brief Constructor
  MeshStreamerBase() = default;
  virtual ~MeshStreamerBase() = default;

  /// @brief Marks these block indices as candidates for streaming (usually
  /// because they have been touched by the reconstruction process).
  /// @param mesh_block_indices The indices of the candidate mesh blocks
  void markIndicesCandidates(const std::vector<Index3D>& mesh_block_indices);

  /// @brief Returns N highest-priority mesh blocks.
  ///
  /// This method modifies the list of tracked indices, removing those which are
  /// returned for streaming.
  /// @param num_mesh_block The number of mesh blocks you want.
  /// @return The list of highest priority mesh block indices.
  std::vector<Index3D> getNMeshBlocks(const int num_mesh_blocks);

  /// @brief Returns highest priority mesh blocks up to N bytes in size
  ///
  /// This method modifies the list of tracked indices, removing those which are
  /// returned for streaming.
  /// @param num_bytes The maximum number of bytes returned
  /// @param mesh_layer The mesh layer which will be streamed. This is used to
  /// check the size (in bytes) of the mesh blocks, so it should be updated
  /// before calling this function.
  /// @return The list of highest priority mesh block indices.
  std::vector<Index3D> getNBytesOfMeshBlocks(const size_t num_bytes,
                                             const MeshLayer& mesh_layer);

  /// @brief Returns highest priority serialized mesh blocks up to N bytes
  ///
  /// @param num_bytes The maximum number of bytes returned
  /// @param mesh_layer Mesh layer to serialize
  /// @param cuda_stream Cuda stream.
  /// @return Serialized mesh containing highest priority mesh blocks
  const std::shared_ptr<const SerializedMesh> getNBytesOfSerializedMeshBlocks(
      const size_t num_bytes, const MeshLayer& mesh_layer,
      const CudaStream cuda_stream);

  /// @brief Returns the number of mesh indices that are currently candidates
  /// for streaming.
  /// @return The number of indices we're keeping track of.
  int numCandidates() const;

  /// @brief Resets the streamer. Clearing all tracked indices.
  void clear();

  // A function which tell the class whether to exclude a block from streaming
  using ExcludeBlockFunctor = std::function<bool(const Index3D&)>;

  // Sets the exclusion functions being used.
  void setExclusionFunctors(
      std::vector<ExcludeBlockFunctor> exclude_block_functors);

 protected:
  // The function which determines a block's priority to be streamed.
  virtual std::vector<float> computePriorities(
      const std::vector<Index3D>& mesh_block_indices) const = 0;

  // Modifies the list of blocks eligible to be streamed in order to remove some
  // blocks before priorities are computed. Internally this function calls the
  // exclusion functors on each block.
  void excludeBlocks(std::vector<Index3D>* mesh_block_indices) const;

  // This struct which indicates whether a block should be streamed
  struct StreamStatus {
    // Should the query block be streamed
    bool should_block_be_streamed = false;
    // Is the query block invalid (and therefore we should stop tracking it)
    bool block_index_invalid = false;
    // Have we reached the streaming limit (and therefore should be stop
    // streaming)
    bool streaming_limit_reached = false;
  };
  // A function that determines if a block should be streamed.
  using StreamStatusFunctor = std::function<StreamStatus(const Index3D&)>;

  // A list of functors for testing blocks to be excluded from streaming
  // altogether.
  std::vector<ExcludeBlockFunctor> exclude_block_functors_;

  /// @brief Common function used by getNMeshBlocks() and getNBytesMeshBlocks().
  /// @param get_stream_status A functor which is called on candidate blocks and
  /// returns the status of the streaming process; for example, if the stream
  /// has reached its bandwidth capacity.
  /// @return The list of mesh blocks to stream.
  std::vector<Index3D> getHighestPriorityMeshBlocks(
      StreamStatusFunctor get_stream_status);

  // This set tracks the mesh blocks which are candidates for streaming but have
  // not yet been streamed.
  Index3DSet mesh_index_set_;

  // Handles serialization of the mesh
  MeshSerializer serializer_;
};

/// @brief A concrete child class of MeshStreamerBase.
///
/// This class impliments a computePriorities() function which prioritizes
/// streaming the oldest MeshBlocks, i.e. the MeshBlocks that have not been
/// streamed to the longest time. Additionally, the streamer adds options for
/// excluding blocks above a certain height and blocks outside a certain radius.
class MeshStreamerOldestBlocks : public MeshStreamerBase {
 public:
  // Parameter defaults
  static constexpr bool kDefaultExcludeBlocksAboveHeight = false;
  static constexpr float kDefaultExclusionHeightM = 2.0;
  static constexpr bool kDefaultExcludeBlocksOutsideRadius = false;
  static constexpr float kDefaultExclusionRadiusM = 10.0;

  /// @brief Constructor
  MeshStreamerOldestBlocks() = default;
  virtual ~MeshStreamerOldestBlocks() = default;

  /// @brief Returns N oldest blocks for publishing
  /// @param num_mesh_block The number of mesh blocks you want.
  /// @param exclusion_center_m Center point for radial exclusion
  /// @return The list of oldest mesh block indices.
  std::vector<Index3D> getNMeshBlocks(
      const int num_mesh_blocks, std::optional<float> block_size = std::nullopt,
      std::optional<Vector3f> exclusion_center_m = std::nullopt);

  /// @brief Return N bytes of oldest blocks for publishing
  /// @param num_bytes The number of bytes of mesh blocks to stream
  /// @param exclusion_center_m Center point for radial exclusion
  /// @return The list of oldest mesh block indices.
  std::vector<Index3D> getNBytesOfMeshBlocks(
      const size_t num_bytes, const MeshLayer& mesh_layer,
      std::optional<Vector3f> exclusion_center_m = std::nullopt);

  /// @brief Returns highest priority serialized mesh blocks up to N bytes
  ///
  /// @param num_bytes The maximum number of bytes returned
  /// @param mesh_layer Mesh layer to serialize
  /// @param exclusion_center_m Center point for radial exclusion
  /// @param cuda_stream Cuda stream.
  /// @return Serialized mesh containing highest priority mesh blocks
  const std::shared_ptr<const SerializedMesh> getNBytesOfSerializedMeshBlocks(
      const size_t num_bytes, const MeshLayer& mesh_layer,
      std::optional<Vector3f> exclusion_center_m, const CudaStream cuda_stream);

  /// Getter
  /// @return Flag indicating whether we should exclude blocks based on height.
  bool exclude_blocks_above_height() const;
  /// Getter
  /// @return The height above which blocks are excluded (if corresponding flag
  /// true). Note that block lower extremities are compared to this number.
  float exclusion_height_m() const;
  /// Getter
  /// @return Flag indicating whether we should exclude blocks outside a radius.
  bool exclude_blocks_outside_radius() const;
  /// Getter
  /// @return The radius at which blocks are excluded from streaming (if
  /// corresponding flag is true). Note that block centers are compared to this
  /// number.
  float exclusion_radius_m() const;

  // Setters
  void exclude_blocks_above_height(bool exclude_blocks_above_height);
  void exclusion_height_m(float exclusion_height_m);
  void exclude_blocks_outside_radius(bool exclude_blocks_outside_radius);
  void exclusion_radius_m(float exclusion_radius_m);

  /// Return the parameter tree.
  /// @return the parameter tree
  virtual parameters::ParameterTreeNode getParameterTree(
      const std::string& name_remap = std::string()) const;

 protected:
  // The method that defines this type of streamer. We compute the priority of
  // blocks as the oldest blocks as being the highest priority.
  virtual std::vector<float> computePriorities(
      const std::vector<Index3D>& mesh_block_indices) const override;
  float computePriority(const Index3D& mesh_block_index) const;

  // Called before blocks are returned to requester. Marks blocks as having been
  // streamed.
  void updateBlocksLastPublishIndex(
      const std::vector<Index3D>& mesh_block_indices);

  // Sets up the exclusion functors for this child class.
  // For this child class we (optionally) set up "blocks above height" and
  // "blocks outside radius" exclusion functors.
  void setupExclusionFunctors(
      const std::optional<float>& block_size = std::nullopt,
      const std::optional<Vector3f>& exclusion_center_m = std::nullopt);
  static ExcludeBlockFunctor getExcludeAboveHeightFunctor(
      const float exclusion_height_m, const float block_size_m);
  static ExcludeBlockFunctor getExcludeOutsideRadiusFunctor(
      const float exclude_blocks_radius_m, const Vector3f& center_m,
      const float block_size_m);

  // Height exclusion params
  bool exclude_blocks_above_height_ = kDefaultExcludeBlocksAboveHeight;
  float exclusion_height_m_ = kDefaultExclusionHeightM;

  // Radius exclusion params
  bool exclude_blocks_outside_radius_ = kDefaultExcludeBlocksOutsideRadius;
  float exclusion_radius_m_ = kDefaultExclusionRadiusM;

  // The counts up with each call to getNMeshBlocks(). It is used to indicate
  // the "when" blocks are returned for streaming.
  int64_t publishing_index_ = 0;

  // This map stores when the mesh block was last published.
  using BlockIndexToLastPublishedIndexMap = Index3DHashMapType<int64_t>::type;
  BlockIndexToLastPublishedIndexMap last_published_map_;
};

}  // namespace nvblox
