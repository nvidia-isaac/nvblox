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


#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/core/accessors.h"
#include "nvblox/core/hash.h"
#include "nvblox/core/layer.h"
#include "nvblox/core/mapper.h"

using namespace nvblox;

float getTruncationDistanceFromLayer(const TsdfLayer& tsdf_layer) {
  float truncation_distance = -1.0f;
  callFunctionOnAllVoxels<TsdfVoxel>(
      tsdf_layer,
      [&truncation_distance](const auto& block_idx, const auto& voxel_idx,
                             const TsdfVoxel* tsdf_voxel) {
        truncation_distance =
            std::max(truncation_distance, tsdf_voxel->distance);
      });
  return truncation_distance;
}

enum class FreespaceState {
  kNearSurface,
  kFreespace,
  kPartiallyObservedFreespace
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  if (argc < 2) {
    LOG(ERROR) << "Pass in path to a map to analyze.";
    return 1;
  }

  std::string map_path = argv[1];
  LOG(INFO) << "Analyzing map at: " << map_path;

  RgbdMapper mapper(map_path, MemoryType::kUnified);
  mapper.mesh_integrator().weld_vertices(false);
  const TsdfLayer& tsdf_layer = mapper.tsdf_layer();

  std::cout << "voxel size: " << mapper.layers().voxel_size() << std::endl;

  // Back calculate truncation distance.
  const float truncation_distance_m =
      getTruncationDistanceFromLayer(mapper.tsdf_layer());
  std::cout << "truncation_distance: " << truncation_distance_m << std::endl;

  // Store block state
  Index3DHashMapType<FreespaceState>::type block_states;

  auto test_freespace = [&](const Index3D& block_index,
                            const Index3D& voxel_index,
                            const TsdfVoxel* tsdf_voxel) {
    // Allocate if required
    FreespaceState* state_ptr;
    auto it = block_states.find(block_index);
    if (it == block_states.end()) {
      auto insert_status =
          block_states.emplace(block_index, FreespaceState::kFreespace);
      state_ptr = &insert_status.first->second;
    } else {
      state_ptr = &it->second;
    }

    // If observed
    if (tsdf_voxel->weight > 0.0f) {
      // If in truncation band
      constexpr float kEps = 0.0005;
      if (std::abs(tsdf_voxel->distance) < (truncation_distance_m - kEps)) {
        *state_ptr = FreespaceState::kNearSurface;
      }
      // If not in the truncation band
      // Leave what it is (starts as freespace)
    }
    // If not observed
    else {
      if (*state_ptr == FreespaceState::kFreespace) {
        *state_ptr = FreespaceState::kPartiallyObservedFreespace;
      }
    }
  };

  callFunctionOnAllVoxels<TsdfVoxel>(tsdf_layer, test_freespace);

  int num_freespace = 0;
  int num_near_surface = 0;
  int num_partially_observed_freespace = 0;
  for (const auto index_state_pair : block_states) {
    const FreespaceState state = index_state_pair.second;
    if (state == FreespaceState::kNearSurface) {
      num_near_surface++;
    } else if (state == FreespaceState::kFreespace) {
      num_freespace++;
    } else {
      num_partially_observed_freespace++;
    }
  }

  std::cout << "num_blocks: " << tsdf_layer.numAllocatedBlocks() << std::endl;
  std::cout << "num_freespace: " << num_freespace << std::endl;
  std::cout << "num_near_surface: " << num_near_surface << std::endl;
  std::cout << "num_partially_observed_freespace: "
            << num_partially_observed_freespace << std::endl;

  std::cout << "freespace ratio: "
            << static_cast<float>(num_freespace) /
                   tsdf_layer.numAllocatedBlocks()
            << std::endl;
  std::cout << "(partially observed) freespace ratio: "
            << static_cast<float>(num_freespace +
                                  num_partially_observed_freespace) /
                   tsdf_layer.numAllocatedBlocks()
            << std::endl;
}