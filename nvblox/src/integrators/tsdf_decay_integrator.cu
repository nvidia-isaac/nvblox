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
#include <nvblox/integrators/tsdf_decay_integrator.h>

namespace nvblox {

TsdfDecayIntegrator::TsdfDecayIntegrator(DecayMode decay_mode)
    : DecayIntegratorBase(decay_mode) {
  if (decay_mode == DecayMode::kDecayToDeallocate) {
    set_free_distance_on_decayed(false);
  } else if (decay_mode == DecayMode::kDecayToFree) {
    set_free_distance_on_decayed(true);
  } else {
    LOG(FATAL) << "Decay mode not implemented";
  }
}

float TsdfDecayIntegrator::decay_factor() const { return decay_factor_; }

void TsdfDecayIntegrator::decay_factor(const float value) {
  CHECK_GT(value, 0.0);
  CHECK_LT(value, 1.0);
  decay_factor_ = value;
}

float TsdfDecayIntegrator::decayed_weight_threshold() const {
  return decayed_weight_threshold_;
};

void TsdfDecayIntegrator::decayed_weight_threshold(
    const float decayed_weight_threshold) {
  CHECK_GE(decayed_weight_threshold, 0.f);
  decayed_weight_threshold_ = decayed_weight_threshold;
};

bool TsdfDecayIntegrator::set_free_distance_on_decayed() const {
  return set_free_distance_on_decayed_;
}

void TsdfDecayIntegrator::set_free_distance_on_decayed(
    const bool set_free_distance_on_decayed) {
  set_free_distance_on_decayed_ = set_free_distance_on_decayed;
}

float TsdfDecayIntegrator::free_distance_vox() const {
  return free_distance_vox_;
}

void TsdfDecayIntegrator::free_distance_vox(const float free_distance_vox) {
  CHECK_GT(free_distance_vox, 0.f);
  free_distance_vox_ = free_distance_vox;
}

__global__ void decayTsdfKernel(TsdfBlock** block_ptrs,
                                const float decay_factor,
                                const float decayed_weight_threshold,
                                const bool decay_to_free,
                                const float free_distance_m,
                                bool* is_block_fully_decayed) {
  // A single thread in each block initializes the block-wise fully-decayed flag
  // to true. Later on, if any voxel is not fully decayed it sets false.
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    is_block_fully_decayed[blockIdx.x] = true;
  }
  __syncthreads();

  TsdfVoxel* voxel_ptr =
      &(block_ptrs[blockIdx.x]->voxels[threadIdx.z][threadIdx.y][threadIdx.x]);
  assert(voxel_ptr != nullptr);

  // Load the weight from global memory
  float weight = voxel_ptr->weight;

  // We only touch voxels which are above or equal to the
  // decayed_weight_threshold. We assume voxels below are unobserved.
  constexpr float kEps = 1e-6;
  if (weight < (decayed_weight_threshold - kEps)) {
    return;
  }

  // Decay the voxel.
  weight *= decay_factor;
  weight = fmaxf(weight, decayed_weight_threshold);

  // Write weight out to global memory
  voxel_ptr->weight = weight;

  // If any voxel in the block is not decayed, set the block's decayed
  // status to false. NOTE: There could be more than one thread writing
  // this value, but because all of them write false it is no issue.
  // If voxel *not* fully decayed, indicate block not fully decayed.
  if (voxel_ptr->weight > (decayed_weight_threshold + kEps)) {
    is_block_fully_decayed[blockIdx.x] = false;
  }
  // If voxel *is* fully decayed, update distance to free
  else if (decay_to_free) {
    voxel_ptr->distance = free_distance_m;
  }
}

void TsdfDecayIntegrator::decayImplementationAsync(
    TsdfLayer* layer_ptr, const CudaStream cuda_stream) {
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = allocated_block_ptrs_host_.size();

  const float free_distance_m = free_distance_vox_ * layer_ptr->voxel_size();

  decayTsdfKernel<<<num_thread_blocks, kThreadsPerBlock, 0, cuda_stream>>>(
      allocated_block_ptrs_device_.data(),  // NOLINT
      decay_factor_,                        // NOLINT
      decayed_weight_threshold_,            // NOLINT
      set_free_distance_on_decayed_,        // NOLINT
      free_distance_m,                      // NOLINT
      block_fully_decayed_device_.data()    // NOLINT
  );
  checkCudaErrors(cudaPeekAtLastError());

  if (set_free_distance_on_decayed_ && deallocate_decayed_blocks_) {
    LOG(WARNING) << "Both \"set_free_distance_on_decayed\", and "
                    "\"deallocate_decayed_blocks\" are set true. These flags "
                    "have conflicting effects";
  }
}

parameters::ParameterTreeNode TsdfDecayIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "tsdf_decay_integrator" : name_remap;
  return ParameterTreeNode(name,
                           {ParameterTreeNode("decay_factor:", decay_factor_),
                            DecayIntegratorBase::getParameterTree()});
}

}  // namespace nvblox
