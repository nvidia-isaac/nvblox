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
#include "nvblox/map/blocks_to_update_tracker.h"

namespace nvblox {

/// Safety vent if size is growing too much. This should not happen as long as
/// the indices are consumed.
void clearIfTooLarge(Index3DSet& set, const std::string& name) {
  constexpr size_t kMaxSize = 100'000;
  if (set.size() > kMaxSize) {
    LOG(ERROR) << "BlocksToUpdateTracker: IndexSet " << name
               << " is too large: " << set.size() << " > " << kMaxSize
               << ". This should normally not happen. Clearing the set";
    set.clear();
  }
}

void BlocksToUpdateTracker::addBlocksToUpdate(
    const std::vector<Index3D>& blocks_to_update,
    const std::vector<BlocksToUpdateType>& block_types_to_update) {
  // Lambda to add blocks to update asyncronously.
  auto funct = [this, blocks_to_update, block_types_to_update]() -> void {
    if (block_types_to_update.empty()) {
      // Add blocks to all initialized types
      for (auto& [block_type, state] : states_) {
        state.insertBlocks(blocks_to_update);
        clearIfTooLarge(state.blockSet(), toString(block_type));
      }
    } else {
      // Add only to specified types
      for (const auto& block_type : block_types_to_update) {
        auto state_it = states_.find(block_type);
        // Only add the new blocks if the state for the block type exists.
        // Note: States are created lazily on first call to getBlocksToUpdate
        // for a given block type, not in addBlocksToUpdate. This prevents
        // creating states for types that are never queried.
        if (state_it != states_.end()) {
          state_it->second.insertBlocks(blocks_to_update);
          clearIfTooLarge(state_it->second.blockSet(), toString(block_type));
        }
      }
    }
  };

  future_.wait();
  future_ = std::async(std::launch::async, funct);
}

void BlocksToUpdateTracker::addAllBlocksToUpdate() {
  auto funct = [this]() -> void {
    // Set flag for all initialized types
    for (auto& [_, state] : states_) {
      state.setUpdateAllBlocks();
    }
  };

  future_.wait();
  future_ = std::async(std::launch::async, funct);
}

void BlocksToUpdateTracker::removeClearedBlocksFromTracking(
    const std::vector<Index3D>& blocks_to_remove) {
  // Lambda to remove blocks from tracking asyncronously.
  auto funct = [this, blocks_to_remove]() -> void {
    // Remove from all existing states (don't create new ones)
    // Note: If update_all_blocks is true, individual blocks aren't tracked,
    // so erasing has no effect.
    for (auto& [_, state] : states_) {
      for (const Index3D& idx : blocks_to_remove) {
        state.eraseBlock(idx);
      }
    }
  };

  future_.wait();
  future_ = std::async(std::launch::async, funct);
}

BlocksToUpdateState BlocksToUpdateTracker::getBlocksToUpdate(
    BlocksToUpdateType blocks_to_update_type) {
  future_.wait();

  // Lazy initialization: create state on first access
  auto state_it = states_.find(blocks_to_update_type);
  if (state_it == states_.end()) {
    // Create a new state for the block type if it doesn't exist (lazy
    // initialization).
    auto& state = states_[blocks_to_update_type];
    // First access - we don't know which blocks need updating, so mark all.
    state.setUpdateAllBlocks();
    state_it = states_.find(blocks_to_update_type);
  }

  // Return a copy of the state (it will convert the set to vector on demand)
  return state_it->second;
}

void BlocksToUpdateTracker::markBlocksAsUpdated(
    BlocksToUpdateType blocks_to_update_type) {
  // Lambda to mark blocks as updated asyncronously.
  auto funct = [this, blocks_to_update_type]() -> void {
    auto state_it = states_.find(blocks_to_update_type);
    if (state_it == states_.end()) {
      NVBLOX_ABORT("Attempted to mark blocks as updated for a state (" +
                   toString(blocks_to_update_type) +
                   ") that does not exist. This is unexpected.");
    }
    state_it->second.markBlocksAsUpdated();
  };

  future_.wait();
  future_ = std::async(std::launch::async, funct);
}

}  // namespace nvblox
