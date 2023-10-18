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
#include "nvblox/mesh/mesh_streamer.h"

#include <numeric>

namespace nvblox {

int MeshStreamerBase::numCandidates() const { return mesh_index_set_.size(); }

void MeshStreamerBase::clear() { return mesh_index_set_.clear(); }

void MeshStreamerBase::markIndicesCandidates(
    const std::vector<Index3D>& mesh_block_indices) {
  mesh_index_set_.insert(mesh_block_indices.begin(), mesh_block_indices.end());
}

void MeshStreamerBase::setExclusionFunctors(
    std::vector<ExcludeBlockFunctor> exclude_block_functors) {
  exclude_block_functors_ = exclude_block_functors;
};

std::vector<Index3D> MeshStreamerBase::getNMeshBlocks(
    const int num_mesh_blocks) {
  // Define a functor that counts the number of blocks streamed so far.
  int num_blocks_streamed = 0;
  StreamStatusFunctor stream_n_blocks_functor =
      [&num_blocks_streamed, num_mesh_blocks](const Index3D&) -> StreamStatus {
    // If we have enough bandwidth stream, otherwise finish streaming
    const bool should_block_be_streamed = num_blocks_streamed < num_mesh_blocks;
    if (should_block_be_streamed) {
      ++num_blocks_streamed;
    }
    return {.should_block_be_streamed = should_block_be_streamed,
            .block_index_invalid = false,
            .streaming_limit_reached = !should_block_be_streamed};
  };

  // Stream N highest priority blocks
  return getHighestPriorityMeshBlocks(stream_n_blocks_functor);
}

std::vector<Index3D> MeshStreamerBase::getNBytesOfMeshBlocks(
    const size_t num_bytes, const MeshLayer& mesh_layer) {
  // Define a functor that counts the number of bytes streamed so far.
  size_t num_bytes_streamed = 0;
  StreamStatusFunctor stream_n_bytes_functor =
      [&num_bytes_streamed, num_bytes,
       &mesh_layer](const Index3D& idx) -> StreamStatus {
    StreamStatus status;
    const MeshBlock::ConstPtr mesh_block_ptr = mesh_layer.getBlockAtIndex(idx);
    // If the mesh block has been deallocated, don't stream and indicate invalid
    if (!mesh_block_ptr) {
      return {.should_block_be_streamed = false,
              .block_index_invalid = true,
              .streaming_limit_reached = false};
    }
    // The bytes that would be sent if we sent this block
    num_bytes_streamed += mesh_block_ptr->sizeInBytes();
    // If we have enough bandwidth send, otherwise stop streaming.
    const bool should_stream = num_bytes_streamed < num_bytes;
    return {.should_block_be_streamed = should_stream,
            .block_index_invalid = false,
            .streaming_limit_reached = !should_stream};
  };

  // Stream N highest priority blocks
  return getHighestPriorityMeshBlocks(stream_n_bytes_functor);
}

const std::shared_ptr<const SerializedMesh>
MeshStreamerBase::getNBytesOfSerializedMeshBlocks(
    const size_t num_bytes, const MeshLayer& mesh_layer,
    const CudaStream cuda_stream) {
  return serializer_.serializeMesh(
      mesh_layer, getNBytesOfMeshBlocks(num_bytes, mesh_layer), cuda_stream);
}

std::vector<Index3D> MeshStreamerBase::getHighestPriorityMeshBlocks(
    StreamStatusFunctor get_stream_status) {
  // Convert the set of indices to a vector for index-based random access
  std::vector<Index3D> mesh_index_vec;
  std::copy(mesh_index_set_.begin(), mesh_index_set_.end(),
            std::back_inserter(mesh_index_vec));

  // Clear the set. We'll repopulate it with the un-streamed vertices later
  mesh_index_set_.clear();

  // Before we compute any priorities we can exclude blocks
  excludeBlocks(&mesh_index_vec);

  // Compute the priority of each mesh block
  const std::vector<float> priorities = computePriorities(mesh_index_vec);

  // Sort the priorities
  std::vector<int> indices(priorities.size());
  std::iota(indices.begin(), indices.end(), 0);

  // Sort high to low
  std::sort(indices.begin(), indices.end(),
            [&](const int a, const int b) -> bool {
              return priorities[a] > priorities[b];
            });

  // Split the mesh_index_vec into highest priority for streaming and the lower
  // priority to store in the class for next time.
  std::vector<Index3D> high_priority_blocks;
  int insert_remaining_start_idx = -1;
  for (int i = 0; i < static_cast<int>(mesh_index_vec.size()); i++) {
    // Get the index of the i'th high priority block.
    const int sorted_index = indices[i];
    const Index3D& mesh_block_index = mesh_index_vec[sorted_index];
    // Call out to the functor to see what we should do with this block.
    const StreamStatus stream_status = get_stream_status(mesh_block_index);
    // Stream
    if (stream_status.should_block_be_streamed) {
      high_priority_blocks.push_back(mesh_block_index);
    }
    // Stay - block should not be streamed but *is* valid.
    else if (!stream_status.block_index_invalid) {
      mesh_index_set_.insert(mesh_block_index);
    }
    // Streaming limit reached, stop testing
    if (stream_status.streaming_limit_reached) {
      insert_remaining_start_idx = i + 1;
      break;
    }
  }
  // If we terminated early, add the untested blocks back into the tracking set.
  if (insert_remaining_start_idx > 0) {
    for (size_t i = insert_remaining_start_idx; i < indices.size(); i++) {
      const int sorted_index = indices[i];
      const Index3D mesh_block_index = mesh_index_vec[sorted_index];
      mesh_index_set_.insert(mesh_block_index);
    }
  }
  // Sanity check that the set isn't getting too big.
  // NOTE(alexmillane): This number is approximately 50m*50m*10m at 0.05m
  // voxels. At this point you really should be using radius-based exclusion (in
  // the child class MeshStreamerOldestBlocks).
  constexpr size_t kNumBlocksWarningThreshold = 500000ul;
  if (mesh_index_set_.size() > kNumBlocksWarningThreshold) {
    LOG(WARNING) << "The number of tracked mesh blocks for streaming is "
                    "getting very large: "
                 << mesh_index_set_.size()
                 << ". Consider adding some form of block exclusion.";
  }
  return high_priority_blocks;
}

void MeshStreamerBase::excludeBlocks(
    std::vector<Index3D>* mesh_block_indices) const {
  if (exclude_block_functors_.size() == 0) {
    // The base implementation is to not exclude any blocks.
    return;
  }
  // Exclude blocks not meeting the exclusion functions
  std::vector<Index3D> mesh_index_vec_not_excluded;
  mesh_index_vec_not_excluded.reserve(mesh_block_indices->size());
  for (const Index3D& block_idx : *mesh_block_indices) {
    bool exclude = false;
    for (const ExcludeBlockFunctor& exclude_block_functor :
         exclude_block_functors_) {
      if (exclude_block_functor(block_idx)) {
        // Exclusion function fired, move to the next block.
        exclude = true;
        break;
      };
    }
    // If none of the exclusion functions fire, we add this block
    if (!exclude) {
      mesh_index_vec_not_excluded.push_back(block_idx);
    }
  }
  // Rewrite the block indices
  *mesh_block_indices = std::move(mesh_index_vec_not_excluded);
}

bool MeshStreamerOldestBlocks::exclude_blocks_above_height() const {
  return exclude_blocks_above_height_;
}

void MeshStreamerOldestBlocks::exclude_blocks_above_height(
    bool exclude_blocks_above_height) {
  exclude_blocks_above_height_ = exclude_blocks_above_height;
}

float MeshStreamerOldestBlocks::exclusion_height_m() const {
  return exclusion_height_m_;
}
void MeshStreamerOldestBlocks::exclusion_height_m(float exclusion_height_m) {
  exclusion_height_m_ = exclusion_height_m;
}

bool MeshStreamerOldestBlocks::exclude_blocks_outside_radius() const {
  return exclude_blocks_outside_radius_;
}

void MeshStreamerOldestBlocks::exclude_blocks_outside_radius(
    bool exclude_blocks_outside_radius) {
  exclude_blocks_outside_radius_ = exclude_blocks_outside_radius;
}

float MeshStreamerOldestBlocks::exclusion_radius_m() const {
  return exclusion_radius_m_;
}

void MeshStreamerOldestBlocks::exclusion_radius_m(float exclusion_radius_m) {
  exclusion_radius_m_ = exclusion_radius_m;
}

std::vector<Index3D> MeshStreamerOldestBlocks::getNMeshBlocks(
    const int num_mesh_blocks, std::optional<float> block_size,
    std::optional<Vector3f> exclusion_center_m) {
  // Calls the base class method, after setting up the exclusion functors.
  setupExclusionFunctors(block_size, exclusion_center_m);
  const std::vector<Index3D> mesh_block_indices =
      MeshStreamerBase::getNMeshBlocks(num_mesh_blocks);
  // Mark these blocks as streamed.
  updateBlocksLastPublishIndex(mesh_block_indices);
  return mesh_block_indices;
}

std::vector<Index3D> MeshStreamerOldestBlocks::getNBytesOfMeshBlocks(
    const size_t num_bytes, const MeshLayer& mesh_layer,
    std::optional<Vector3f> exclusion_center_m) {
  // Calls the base class method, after setting up the exclusion functors.
  setupExclusionFunctors(mesh_layer.block_size(), exclusion_center_m);
  const std::vector<Index3D> mesh_block_indices =
      MeshStreamerBase::getNBytesOfMeshBlocks(num_bytes, mesh_layer);
  // Mark these blocks as streamed.
  updateBlocksLastPublishIndex(mesh_block_indices);
  return mesh_block_indices;
}

const std::shared_ptr<const SerializedMesh>
MeshStreamerOldestBlocks::getNBytesOfSerializedMeshBlocks(
    const size_t num_bytes, const MeshLayer& mesh_layer,
    std::optional<Vector3f> exclusion_center_m, const CudaStream cuda_stream) {
  return serializer_.serializeMesh(
      mesh_layer,
      getNBytesOfMeshBlocks(num_bytes, mesh_layer, exclusion_center_m),
      cuda_stream);
}

std::vector<float> MeshStreamerOldestBlocks::computePriorities(
    const std::vector<Index3D>& mesh_block_indices) const {
  std::vector<float> priorities;
  std::transform(mesh_block_indices.begin(), mesh_block_indices.end(),
                 std::back_inserter(priorities),
                 [&](const Index3D& mesh_block_index) {
                   return computePriority(mesh_block_index);
                 });
  return priorities;
}

float MeshStreamerOldestBlocks::computePriority(
    const Index3D& mesh_block_index) const {
  // More recently published blocks should have lower priority, so we negate
  // the last published index (more recent blocks have a higher index value,
  // and therefore lower priority)
  const auto it = last_published_map_.find(mesh_block_index);
  if (it == last_published_map_.end()) {
    return static_cast<float>(std::numeric_limits<int64_t>::max());
  } else {
    return static_cast<float>(-1 * it->second);
  }
}

void MeshStreamerOldestBlocks::updateBlocksLastPublishIndex(
    const std::vector<Index3D>& mesh_block_indices) {
  // Go through the blocks and mark them as having been streamed.
  for (const Index3D& block_idx : mesh_block_indices) {
    // Insert a new publishing index or update the old one.
    last_published_map_[block_idx] = publishing_index_;
  }
  ++publishing_index_;
  CHECK_LT(publishing_index_, std::numeric_limits<int64_t>::max());
}

void MeshStreamerOldestBlocks::setupExclusionFunctors(
    const std::optional<float>& block_size,
    const std::optional<Vector3f>& exclusion_center_m) {
  // If requested exclude blocks
  std::vector<ExcludeBlockFunctor> exclusion_functors;
  // Exclude based on height
  if (exclude_blocks_above_height_) {
    if (block_size.has_value()) {
      exclusion_functors.push_back(
          getExcludeAboveHeightFunctor(exclusion_height_m_, *block_size));
    } else {
      LOG(WARNING) << "You requested height based exclusion of blocks but "
                      "didn't pass a block size. Skipping this exclusion.";
    }
  }
  // Exclude based on radius
  if (exclude_blocks_outside_radius_) {
    if (block_size.has_value() && exclusion_center_m.has_value()) {
      exclusion_functors.push_back(getExcludeOutsideRadiusFunctor(
          exclusion_radius_m_, *exclusion_center_m, *block_size));
    } else {
      LOG(WARNING)
          << "You requested radius based exclusion of blocks but "
             "didn't pass a block size or center. Skipping this exclusion.";
    }
  }
  setExclusionFunctors(exclusion_functors);
}

MeshStreamerOldestBlocks::ExcludeBlockFunctor
MeshStreamerOldestBlocks::getExcludeAboveHeightFunctor(
    const float exclusion_height_m, const float block_size_m) {
  // Create a functor which returns true if a blocks minimum height is above
  // a limit.
  return [exclusion_height_m, block_size_m](const Index3D& idx) -> bool {
    // Exclude block if it's low corner/plane is above the exclusion
    // limit
    const float low_z_m = static_cast<float>(idx.z()) * block_size_m;
    return low_z_m > exclusion_height_m;
  };
}

MeshStreamerOldestBlocks::ExcludeBlockFunctor
MeshStreamerOldestBlocks::getExcludeOutsideRadiusFunctor(
    const float exclude_blocks_radius_m, const Vector3f& center_m,
    const float block_size_m) {
  // Square the radius outside
  const float exclude_blocks_radius_squared_m2 =
      exclude_blocks_radius_m * exclude_blocks_radius_m;
  // Create a functor which returns true if the block center is a greater radius
  // from the passed center.
  return [exclude_blocks_radius_squared_m2, center_m,
          block_size_m](const Index3D& idx) -> bool {
    // Calculate the blocks center position
    const Vector3f block_center =
        getCenterPositionFromBlockIndex(block_size_m, idx);
    const float block_radius_squared_m2 =
        (block_center - center_m).squaredNorm();
    return block_radius_squared_m2 > exclude_blocks_radius_squared_m2;
  };
}

parameters::ParameterTreeNode MeshStreamerOldestBlocks::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name = (name_remap.empty()) ? "mesh_streamer" : name_remap;
  return ParameterTreeNode(
      name, {ParameterTreeNode("exclude_blocks_above_height:",
                               exclude_blocks_above_height_),
             ParameterTreeNode("exclusion_height_m:", exclusion_height_m_),
             ParameterTreeNode("exclude_blocks_outside_radius:",
                               exclude_blocks_outside_radius_),
             ParameterTreeNode("exclusion_radius_m:", exclusion_radius_m_)});
}

};  // namespace nvblox
