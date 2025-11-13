/*
Copyright 2025 NVIDIA CORPORATION

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

#include <future>
#include <map>
#include <vector>

#include "nvblox/core/hash.h"
#include "nvblox/core/internal/error_check.h"
#include "nvblox/core/types.h"

namespace nvblox {

/// @brief Types of blocks being tracked.
enum class BlocksToUpdateType {
  kEsdf,
  kColorMesh,
  kFeatureMesh,
  kFreespace,
  kLayerStreamer
};

/// @brief Convert BlocksToUpdateType to string for logging.
inline std::string toString(BlocksToUpdateType type) {
  switch (type) {
    case BlocksToUpdateType::kEsdf:
      return "esdf";
    case BlocksToUpdateType::kColorMesh:
      return "color_mesh";
    case BlocksToUpdateType::kFeatureMesh:
      return "feature_mesh";
    case BlocksToUpdateType::kFreespace:
      return "freespace";
    case BlocksToUpdateType::kLayerStreamer:
      return "layer_streamer";
    default:
      LOG(FATAL) << "Not implemented";
      return "";
  }
}

/// @brief Tracking state for blocks that need updating (for a single block
/// type).
/// @note Makes sure that if update_all_blocks is true, then blocks_set is
/// empty.
struct BlocksToUpdateState {
  BlocksToUpdateState() : update_all_blocks_(false) {}

  // Query interface
  bool updateAll() const { return update_all_blocks_; }

  std::vector<Index3D> blocks() const {
    // The user should always query updateAll() before querying individual
    // blocks().
    NVBLOX_CHECK(
        !update_all_blocks_,
        "Querying individual blocks with updateAll()=true is not allowed");
    return std::vector<Index3D>(blocks_set_.begin(), blocks_set_.end());
  }

  // Mutation interface
  void insertBlocks(const std::vector<Index3D>& blocks) {
    // If update_all_blocks is true, no need to track individual blocks.
    if (!update_all_blocks_) {
      blocks_set_.insert(blocks.begin(), blocks.end());
    }
  }

  void setUpdateAllBlocks() {
    update_all_blocks_ = true;
    blocks_set_.clear();
  }

  void markBlocksAsUpdated() {
    update_all_blocks_ = false;
    blocks_set_.clear();
  }

  void eraseBlock(const Index3D& block) { blocks_set_.erase(block); }

  Index3DSet& blockSet() { return blocks_set_; }

 private:
  bool update_all_blocks_;
  Index3DSet blocks_set_;
};

/// @brief Class to keep track of blocks that need to be updated per block type.
/// @note The tracking is implemented with lazy initialization.
///       This means the tracking state for a block type is only
///       started/initialized after the first call to getBlocksToUpdate for that
///       block type. The first call to getBlocksToUpdate for a block type will
///       return update_all_blocks=true to ensure no blocks are missed.
class BlocksToUpdateTracker {
 public:
  BlocksToUpdateTracker() = default;

  /// @brief Adding blocks that need an update.
  /// @param blocks_to_update Vector of block indices that need an update.
  /// @param block_types_to_update Which tracking types to add these blocks to.
  ///        If empty, adds to all initialized types.
  void addBlocksToUpdate(
      const std::vector<Index3D>& blocks_to_update,
      const std::vector<BlocksToUpdateType>& block_types_to_update = {});

  /// @brief Add all blocks to update for all initialized types.
  void addAllBlocksToUpdate();

  /// @brief Remove blocks that have been cleared/deallocated from tracking.
  /// @param blocks_to_remove Vector of block indices that were deallocated from
  ///        the layers and should be removed from all tracking sets.
  /// @note This is called when blocks are cleared from layers. Blocks marked
  ///       with update_all_blocks=true don't track individual blocks, so
  ///       removal has no effect in that case.
  void removeClearedBlocksFromTracking(
      const std::vector<Index3D>& blocks_to_remove);

  /// @brief Get the blocks that need update. If a tracking type is not
  ///        initialized, it will be initialized and the function will
  ///        return update_all_blocks=true (lazy initialization).
  /// @param blocks_to_update_type The type of blocks you want to get.
  /// @return BlocksToUpdateState with update_all_blocks flag or specific block
  /// list.
  BlocksToUpdateState getBlocksToUpdate(
      BlocksToUpdateType blocks_to_update_type);

  /// @brief Mark all blocks of a block type to be updated.
  /// @param blocks_to_update_type The type of blocks that got updated.
  void markBlocksAsUpdated(BlocksToUpdateType blocks_to_update_type);

 private:
  /// Map tracking blocks to update for each type (created on-demand)
  std::map<BlocksToUpdateType, BlocksToUpdateState> states_;

  // Object to synchronize async functions (initialize to valid)
  mutable std::future<void> future_ = std::async(std::launch::async, []() {});
};

}  // namespace nvblox
