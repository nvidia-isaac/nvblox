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
#include <fstream>
#include <iostream>

#include <gflags/gflags.h>
#include <glog/logging.h>

#include "nvblox/core/camera.h"
#include "nvblox/core/common_names.h"
#include "nvblox/core/cuda/warmup.h"
#include "nvblox/core/types.h"
#include "nvblox/datasets/parse_3dmatch.h"
#include "nvblox/integrators/integrators_common.h"
#include "nvblox/integrators/projective_tsdf_integrator.h"
#include "nvblox/utils/timing.h"

DECLARE_bool(alsologtostderr);

using namespace nvblox;

// Just a class so we can acces integrator internals
class ProjectiveTsdfIntegratorExperiment : public ProjectiveTsdfIntegrator {
 public:
  ProjectiveTsdfIntegratorExperiment() : ProjectiveTsdfIntegrator() {}
  virtual ~ProjectiveTsdfIntegratorExperiment(){};

  // Expose this publically
  void updateBlocks(const std::vector<Index3D>& block_indices,
                    const DepthImage& depth_frame, const Transform& T_L_C,
                    const Camera& camera, const float truncation_distance_m,
                    TsdfLayer* layer) {
    ProjectiveTsdfIntegrator::updateBlocks(block_indices, depth_frame, T_L_C,
                                           camera, truncation_distance_m,
                                           layer);
  }
};

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  const std::string dataset_base_path = "../../../tests/data/3dmatch";
  constexpr int kSeqNum = 1;

  constexpr float kVoxelSize = 0.05;
  TsdfLayer tsdf_layer(kVoxelSize, MemoryType::kDevice);

  ProjectiveTsdfIntegratorExperiment tsdf_integrator;

  const unsigned int frustum_raycast_subsampling_rate = 4;
  tsdf_integrator.frustum_calculator().raycast_subsampling_factor(
      frustum_raycast_subsampling_rate);

  // Update identified blocks (many times)
  constexpr int kNumIntegrations = 1000;
  for (int i = 0; i < kNumIntegrations; i++) {
    // Load images
    auto image_loader_ptr = datasets::threedmatch::createDepthImageLoader(
        dataset_base_path, kSeqNum);

    DepthImage depth_frame;
    CHECK(image_loader_ptr->getNextImage(&depth_frame));

    Eigen::Matrix3f camera_intrinsics;
    CHECK(datasets::threedmatch::parseCameraFromFile(
        datasets::threedmatch::getPathForCameraIntrinsics(dataset_base_path),
        &camera_intrinsics));
    const auto camera = Camera::fromIntrinsicsMatrix(
        camera_intrinsics, depth_frame.width(), depth_frame.height());

    Transform T_L_C;
    CHECK(datasets::threedmatch::parsePoseFromFile(
        datasets::threedmatch::getPathForFramePose(dataset_base_path, kSeqNum,
                                                   0),
        &T_L_C));

    // Identify blocks we can (potentially) see (CPU)
    timing::Timer blocks_in_view_timer("tsdf/integrate/get_blocks_in_view");
    const std::vector<Index3D> block_indices =
        tsdf_integrator.getBlocksInViewUsingRaycasting(
            depth_frame, T_L_C, camera, tsdf_layer.block_size());
    blocks_in_view_timer.Stop();

    // Allocate blocks (CPU)
    timing::Timer allocate_blocks_timer("tsdf/integrate/allocate_blocks");
    allocateBlocksWhereRequired(block_indices, &tsdf_layer);
    allocate_blocks_timer.Stop();

    timing::Timer update_blocks_timer("tsdf/integrate/update_blocks");
    const float truncation_distance_m =
        tsdf_integrator.truncation_distance_vox() * kVoxelSize;
    tsdf_integrator.updateBlocks(block_indices, depth_frame, T_L_C, camera,
                                 truncation_distance_m, &tsdf_layer);
    update_blocks_timer.Stop();

    // Reset the layer such that we do TsdfBlock allocation.
    // tsdf_layer.clear();
  }

  std::cout << timing::Timing::Print() << std::endl;

  return 0;
}