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
#include <gtest/gtest.h>

#include <algorithm>

#include "nvblox/core/types.h"
#include "nvblox/map/blocks_to_update_tracker.h"

using namespace nvblox;

class BlocksToUpdateTrackerTest : public ::testing::Test {
 protected:
  BlocksToUpdateTrackerTest() {}

  // Helper to check if a vector contains an element
  bool containsBlock(const std::vector<Index3D>& blocks, const Index3D& block) {
    return std::find(blocks.begin(), blocks.end(), block) != blocks.end();
  }
};

TEST_F(BlocksToUpdateTrackerTest, addBlocksToUpdate) {
  BlocksToUpdateTracker tracker;

  // Add some blocks before any type is initialized (goes nowhere)
  std::vector<Index3D> blocks_to_add = {Index3D(0, 0, 0), Index3D(1, 1, 1),
                                        Index3D(2, 2, 2)};
  tracker.addBlocksToUpdate(blocks_to_add);

  // First access to kEsdf - returns update_all_blocks=true (lazy init)
  BlocksToUpdateState esdf_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(esdf_result.updateAll());

  // Mark as updated to transition to incremental mode
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  // Now ESDF is initialized and cleared, add more blocks
  std::vector<Index3D> more_blocks = {Index3D(3, 3, 3), Index3D(4, 4, 4)};
  tracker.addBlocksToUpdate(more_blocks);

  // Get ESDF blocks again - should have specific blocks now
  BlocksToUpdateState esdf_result2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_result2.updateAll());
  EXPECT_EQ(esdf_result2.blocks().size(), 2);
  EXPECT_TRUE(containsBlock(esdf_result2.blocks(), Index3D(3, 3, 3)));
  EXPECT_TRUE(containsBlock(esdf_result2.blocks(), Index3D(4, 4, 4)));

  // First access to ColorMesh - returns update_all_blocks=true
  BlocksToUpdateState mesh_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_TRUE(mesh_result.updateAll());

  // Mark mesh as updated
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kColorMesh);

  // Add more blocks - should go to both ESDF and ColorMesh now
  std::vector<Index3D> final_blocks = {Index3D(5, 5, 5)};
  tracker.addBlocksToUpdate(final_blocks);

  BlocksToUpdateState esdf_result3 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_result3.updateAll());
  EXPECT_EQ(esdf_result3.blocks().size(), 3);
  EXPECT_TRUE(containsBlock(esdf_result3.blocks(), Index3D(5, 5, 5)));

  BlocksToUpdateState mesh_result2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_FALSE(mesh_result2.updateAll());
  EXPECT_EQ(mesh_result2.blocks().size(), 1);
  EXPECT_TRUE(containsBlock(mesh_result2.blocks(), Index3D(5, 5, 5)));
}

TEST_F(BlocksToUpdateTrackerTest, SelectiveTypeUpdates) {
  BlocksToUpdateTracker tracker;

  // Add blocks only to ColorMesh (but ColorMesh not initialized yet)
  std::vector<Index3D> mesh_blocks = {Index3D(0, 0, 0), Index3D(1, 1, 1)};
  tracker.addBlocksToUpdate(mesh_blocks, {BlocksToUpdateType::kColorMesh});

  // ColorMesh gets lazy init with update_all_blocks=true on first access
  BlocksToUpdateState mesh_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_TRUE(mesh_result.updateAll());

  // Mark as updated to transition to incremental mode
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kColorMesh);

  // Add more blocks only to ColorMesh (now initialized and in incremental mode)
  std::vector<Index3D> more_mesh_blocks = {Index3D(2, 2, 2)};
  tracker.addBlocksToUpdate(more_mesh_blocks, {BlocksToUpdateType::kColorMesh});

  BlocksToUpdateState mesh_result2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_FALSE(mesh_result2.updateAll());
  EXPECT_EQ(mesh_result2.blocks().size(), 1);
  EXPECT_TRUE(containsBlock(mesh_result2.blocks(), Index3D(2, 2, 2)));

  // ESDF was never accessed, so first access returns update_all_blocks=true
  BlocksToUpdateState esdf_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(esdf_result.updateAll());

  // Verify ColorMesh blocks weren't added to ESDF
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);
  BlocksToUpdateState esdf_result2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_result2.updateAll());
  EXPECT_TRUE(esdf_result2.blocks().empty());  // Should be empty
}

TEST_F(BlocksToUpdateTrackerTest, AddAllBlocksToUpdate) {
  BlocksToUpdateTracker tracker;

  // Test 1: addAllBlocksToUpdate on empty tracker
  tracker.addAllBlocksToUpdate();

  // First access still returns update_all_blocks=true (lazy init already set
  // it)
  BlocksToUpdateState esdf_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(esdf_result.updateAll());

  // Test 2: Combined lifecycle test - incremental → addAll → incremental →
  // addAll
  BlocksToUpdateTracker tracker2;

  // Initialize by accessing first
  BlocksToUpdateState init_result =
      tracker2.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(init_result.updateAll());

  // Mark as updated to transition to incremental mode
  tracker2.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  // Add blocks (now in incremental mode)
  tracker2.addBlocksToUpdate({Index3D(2, 2, 2)});
  BlocksToUpdateState result1 =
      tracker2.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(result1.updateAll());
  EXPECT_EQ(result1.blocks().size(), 1);

  // Call addAllBlocksToUpdate - should set flag and clear blocks
  tracker2.addAllBlocksToUpdate();

  BlocksToUpdateState result2 =
      tracker2.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(result2.updateAll());

  // Add blocks after addAllBlocksToUpdate - should be ignored (flag still true)
  tracker2.addBlocksToUpdate({Index3D(5, 5, 5)});
  BlocksToUpdateState result3 =
      tracker2.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(result3.updateAll());  // Still true

  // Mark as updated again to clear flag
  tracker2.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  // Now blocks should be added
  tracker2.addBlocksToUpdate({Index3D(10, 10, 10)});
  BlocksToUpdateState result4 =
      tracker2.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(result4.updateAll());
  EXPECT_EQ(result4.blocks().size(), 1);
  EXPECT_TRUE(containsBlock(result4.blocks(), Index3D(10, 10, 10)));
}

TEST_F(BlocksToUpdateTrackerTest, RemoveClearedBlocksFromTracking) {
  BlocksToUpdateTracker tracker;

  // Initialize ESDF
  BlocksToUpdateState init_result =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(init_result.updateAll());

  // Mark as updated to transition to incremental mode
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  // Add more blocks
  tracker.addBlocksToUpdate(
      {Index3D(10, 10, 10), Index3D(11, 11, 11), Index3D(12, 12, 12)});

  BlocksToUpdateState result1 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(result1.updateAll());
  EXPECT_EQ(result1.blocks().size(), 3);

  // Remove some blocks
  tracker.removeClearedBlocksFromTracking({Index3D(10, 10, 10)});

  BlocksToUpdateState result2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(result2.updateAll());
  EXPECT_EQ(result2.blocks().size(), 2);
  EXPECT_FALSE(containsBlock(result2.blocks(), Index3D(10, 10, 10)));
  EXPECT_TRUE(containsBlock(result2.blocks(), Index3D(11, 11, 11)));
  EXPECT_TRUE(containsBlock(result2.blocks(), Index3D(12, 12, 12)));

  // Remove multiple blocks
  tracker.removeClearedBlocksFromTracking(
      {Index3D(11, 11, 11), Index3D(12, 12, 12)});

  BlocksToUpdateState result3 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(result3.updateAll());
  EXPECT_TRUE(result3.blocks().empty());

  // Test removal when update_all_blocks is true (should have no effect on flag)
  tracker.addBlocksToUpdate({Index3D(20, 20, 20)});
  tracker.addAllBlocksToUpdate();

  BlocksToUpdateState result4 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(result4.updateAll());

  tracker.removeClearedBlocksFromTracking({Index3D(20, 20, 20)});

  BlocksToUpdateState result5 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_TRUE(result5.updateAll());  // Still true
}

TEST_F(BlocksToUpdateTrackerTest, MarkBlocksAsUpdated) {
  BlocksToUpdateTracker tracker;

  // Add blocks to multiple types
  tracker.addBlocksToUpdate(
      {Index3D(0, 0, 0), Index3D(1, 1, 1)},
      {BlocksToUpdateType::kEsdf, BlocksToUpdateType::kColorMesh});

  // Access both types to initialize
  BlocksToUpdateState esdf_init =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  BlocksToUpdateState mesh_init =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_TRUE(esdf_init.updateAll());
  EXPECT_TRUE(mesh_init.updateAll());

  // Mark ESDF as updated
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);

  BlocksToUpdateState esdf_after_mark =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_after_mark.updateAll());
  EXPECT_TRUE(esdf_after_mark.blocks().empty());

  // ColorMesh should still have update_all_blocks=true
  BlocksToUpdateState mesh_after_mark =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_TRUE(mesh_after_mark.updateAll());

  // Add new blocks - should go to both types
  tracker.addBlocksToUpdate({Index3D(5, 5, 5)});

  BlocksToUpdateState esdf_with_blocks =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_with_blocks.updateAll());
  EXPECT_EQ(esdf_with_blocks.blocks().size(), 1);
  EXPECT_TRUE(containsBlock(esdf_with_blocks.blocks(), Index3D(5, 5, 5)));

  // ColorMesh still has update_all_blocks=true, so blocks not tracked
  BlocksToUpdateState mesh_still_all =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_TRUE(mesh_still_all.updateAll());

  // Mark ColorMesh as updated
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kColorMesh);

  BlocksToUpdateState mesh_after_mark2 =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_FALSE(mesh_after_mark2.updateAll());
  EXPECT_TRUE(mesh_after_mark2.blocks().empty());

  // Add blocks and verify both types get them now
  tracker.addBlocksToUpdate({Index3D(10, 10, 10), Index3D(11, 11, 11)});

  BlocksToUpdateState esdf_final =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  EXPECT_FALSE(esdf_final.updateAll());
  EXPECT_EQ(esdf_final.blocks().size(), 3);  // 5,5,5 + 10,10,10 + 11,11,11

  BlocksToUpdateState mesh_final =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_FALSE(mesh_final.updateAll());
  EXPECT_EQ(mesh_final.blocks().size(), 2);  // 10,10,10 + 11,11,11

  // Mark both as updated
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kEsdf);
  tracker.markBlocksAsUpdated(BlocksToUpdateType::kColorMesh);

  BlocksToUpdateState esdf_cleared =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kEsdf);
  BlocksToUpdateState mesh_cleared =
      tracker.getBlocksToUpdate(BlocksToUpdateType::kColorMesh);
  EXPECT_FALSE(esdf_cleared.updateAll());
  EXPECT_TRUE(esdf_cleared.blocks().empty());
  EXPECT_FALSE(mesh_cleared.updateAll());
  EXPECT_TRUE(mesh_cleared.blocks().empty());
}

int main(int argc, char** argv) {
  FLAGS_alsologtostderr = true;
  google::InitGoogleLogging(argv[0]);
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
