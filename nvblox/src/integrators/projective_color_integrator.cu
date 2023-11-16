/*
Copyright 2022-2023 NVIDIA CORPORATION

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

#include "nvblox/core/color.h"
#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/projective_color_integrator.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

ProjectiveColorIntegrator::ProjectiveColorIntegrator()
    : ProjectiveColorIntegrator(std::make_shared<CudaStreamOwning>()) {}

ProjectiveColorIntegrator::ProjectiveColorIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : ProjectiveIntegrator<ColorVoxel>(cuda_stream),
      update_functor_host_ptr_(
          make_unified<UpdateColorVoxelFunctor>(MemoryType::kHost)),
      sphere_tracer_(cuda_stream),
      synthetic_depth_image_(MemoryType::kDevice) {
  sphere_tracer_.maximum_ray_length_m(max_integration_distance_m_);
}

// NOTE(dtingdahl): We can't default this in the header file because to the
// unified_ptr to a forward declared type. The type has to be defined where
// the destructor is.
ProjectiveColorIntegrator::~ProjectiveColorIntegrator() = default;

void ProjectiveColorIntegrator::integrateFrame(
    const ColorImage& color_frame, const Transform& T_L_C, const Camera& camera,
    const TsdfLayer& tsdf_layer, ColorLayer* color_layer,
    std::vector<Index3D>* updated_blocks) {
  timing::Timer color_timer("color/integrate");
  CHECK_NOTNULL(color_layer);
  CHECK_EQ(tsdf_layer.block_size(), color_layer->block_size());

  // Metric truncation distance for this layer
  const float voxel_size =
      color_layer->block_size() / VoxelBlock<bool>::kVoxelsPerSide;
  const float truncation_distance_m =
      truncation_distance_vox_ * tsdf_layer.voxel_size();

  // TODO(alexmillane): This order of operations could be improved here. We
  // could:
  // - Create synthetic depth *first*
  // - Then use the depth image to gets block in view, as we do in the
  //   TSDFIntegrator.
  // - We could add an option to the view calculator to only return blocks in
  //   the truncation band.
  // - We could then remove the kernel below for reduce blocks to those in the
  // truncation band.

  timing::Timer blocks_in_view_timer("color/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices = view_calculator_.getBlocksInViewPlanes(
      T_L_C, camera, color_layer->block_size(),
      max_integration_distance_m_ + truncation_distance_m);
  blocks_in_view_timer.Stop();

  // Check which of these blocks are:
  // - Allocated in the TSDF, and
  // - have at least a single voxel within the truncation band
  // This is because:
  // - We don't allocate new geometry here, we just color existing geometry
  // - We don't color freespace.
  timing::Timer blocks_in_band_timer(
      "color/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  if (block_indices.empty()) {
    return;
  }
  blocks_in_band_timer.Stop();

  // Allocate blocks (CPU)
  // We allocate color blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer("color/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, color_layer, *cuda_stream_);
  allocate_blocks_timer.Stop();

  // Create a synthetic depth image
  timing::Timer sphere_trace_timer("color/integrate/sphere_trace");
  sphere_tracer_.renderImageOnGPU(
      camera, T_L_C, tsdf_layer, truncation_distance_m, &synthetic_depth_image_,
      MemoryType::kDevice, sphere_tracing_ray_subsampling_factor_);
  sphere_trace_timer.Stop();

  timing::Timer transfer_blocks_timer("color/integrate/transfer_blocks");
  transferBlockPointersToDevice<ColorBlock>(block_indices, *cuda_stream_,
                                            color_layer, &block_ptrs_host_,
                                            &block_ptrs_device_);
  transferBlocksIndicesToDevice(block_indices, *cuda_stream_,
                                &block_indices_host_, &block_indices_device_);

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Move the functor to the GPU
  unified_ptr<UpdateColorVoxelFunctor> update_functor_device =
      getColorUpdateFunctorOnDevice(tsdf_layer.voxel_size());
  transfer_blocks_timer.Stop();

  // Calling the GPU to do the updates
  timing::Timer update_blocks_timer("color/integrate/update_blocks");
  integrateBlocks(synthetic_depth_image_, color_frame, T_C_L, camera,
                  update_functor_device.get(), color_layer);

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

std::string ProjectiveColorIntegrator::getIntegratorName() const {
  return "color";
}

void ProjectiveColorIntegrator::sphere_tracing_ray_subsampling_factor(
    int sphere_tracing_ray_subsampling_factor) {
  CHECK_GT(sphere_tracing_ray_subsampling_factor, 0);
  sphere_tracing_ray_subsampling_factor_ =
      sphere_tracing_ray_subsampling_factor;
}

int ProjectiveColorIntegrator::sphere_tracing_ray_subsampling_factor() const {
  return sphere_tracing_ray_subsampling_factor_;
}

float ProjectiveColorIntegrator::max_weight() const { return max_weight_; }

void ProjectiveColorIntegrator::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

float ProjectiveColorIntegrator::get_truncation_distance_m(
    float voxel_size) const {
  return truncation_distance_vox_ * voxel_size;
}

WeightingFunctionType ProjectiveColorIntegrator::weighting_function_type()
    const {
  return weighting_function_type_;
}

void ProjectiveColorIntegrator::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

const ViewCalculator& ProjectiveColorIntegrator::view_calculator() const {
  return view_calculator_;
}

/// Returns the object used to calculate the blocks in camera views.
ViewCalculator& ProjectiveColorIntegrator::view_calculator() {
  return view_calculator_;
}

parameters::ParameterTreeNode ProjectiveColorIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "projective_color_integrator" : name_remap;
  // NOTE(alexmillane): Wrapping our weighting function to_string version in the
  // std::function for passing to the parameter tree node constructor because it
  // seems to have trouble with template deduction.
  std::function<std::string(const WeightingFunctionType&)>
      weighting_function_to_string =
          [](const WeightingFunctionType& w) { return to_string(w); };
  return ParameterTreeNode(
      name, {
                ParameterTreeNode("sphere_tracing_ray_subsampling_factor:",
                                  sphere_tracing_ray_subsampling_factor_),
                ParameterTreeNode("max_weight:", max_weight_),
                ParameterTreeNode(
                    "weighting_function_type:", weighting_function_type_,
                    weighting_function_to_string),
                ProjectiveIntegrator<ColorVoxel>::getParameterTree(),
                view_calculator_.getParameterTree(),
            });
}

__device__ inline Color blendTwoColors(const Color& first_color,
                                       float first_weight,
                                       const Color& second_color,
                                       float second_weight) {
  float total_weight = first_weight + second_weight;

  first_weight /= total_weight;
  second_weight /= total_weight;

  Color new_color;
  new_color.r = static_cast<uint8_t>(std::round(
      first_color.r * first_weight + second_color.r * second_weight));
  new_color.g = static_cast<uint8_t>(std::round(
      first_color.g * first_weight + second_color.g * second_weight));
  new_color.b = static_cast<uint8_t>(std::round(
      first_color.b * first_weight + second_color.b * second_weight));

  return new_color;
}

struct UpdateColorVoxelFunctor {
  __host__ __device__ UpdateColorVoxelFunctor() = default;
  __host__ __device__ ~UpdateColorVoxelFunctor() = default;

  __device__ bool operator()(const float measured_depth_m,
                             const float voxel_depth_m,
                             const Color& color_measured,
                             ColorVoxel* voxel_ptr) {
    // Read CURRENT voxel values (from global GPU memory)
    const Color voxel_color_current = voxel_ptr->color;
    const float voxel_weight_current = voxel_ptr->weight;
    // Fuse
    const float measurement_weight = weighting_function_(
        measured_depth_m, voxel_depth_m, truncation_distance_m_);
    const Color fused_color =
        blendTwoColors(voxel_color_current, voxel_weight_current,
                       color_measured, measurement_weight);
    const float weight =
        fmin(measurement_weight + voxel_weight_current, max_weight_);
    // Write NEW voxel values (to global GPU memory)
    voxel_ptr->color = fused_color;
    voxel_ptr->weight = weight;
    return true;
  }
  WeightingFunction weighting_function_ = WeightingFunction(
      ProjectiveColorIntegrator::kDefaultWeightingFunctionType);
  float truncation_distance_m_ = 0.2f;
  float max_weight_ = ProjectiveColorIntegrator::kDefaultMaxWeight;
};

unified_ptr<UpdateColorVoxelFunctor>
ProjectiveColorIntegrator::getColorUpdateFunctorOnDevice(float voxel_size) {
  // Set the update function params
  // NOTE(alex.millane): We do this with every frame integration to avoid
  // bug-prone logic for detecting when params have changed etc.
  CHECK(update_functor_host_ptr_ != nullptr);
  update_functor_host_ptr_->max_weight_ = max_weight();
  update_functor_host_ptr_->truncation_distance_m_ =
      get_truncation_distance_m(voxel_size);
  update_functor_host_ptr_->weighting_function_ =
      WeightingFunction(weighting_function_type_);
  // Transfer to the device
  return update_functor_host_ptr_.cloneAsync(MemoryType::kDevice,
                                             *cuda_stream_);
}

__device__ inline bool updateVoxel(const Color color_measured,
                                   const float measured_depth_m,
                                   const float voxel_depth_m,
                                   const float max_weight,
                                   const float truncation_distance_m,
                                   const WeightingFunction& weighting_function,
                                   ColorVoxel* voxel_ptr) {
  // Read CURRENT voxel values (from global GPU memory)
  const Color voxel_color_current = voxel_ptr->color;
  const float voxel_weight_current = voxel_ptr->weight;
  // Fuse
  const float measurement_weight = weighting_function(
      measured_depth_m, voxel_depth_m, truncation_distance_m);
  const Color fused_color =
      blendTwoColors(voxel_color_current, voxel_weight_current, color_measured,
                     measurement_weight);
  const float weight =
      fmin(measurement_weight + voxel_weight_current, max_weight);
  // Write NEW voxel values (to global GPU memory)
  voxel_ptr->color = fused_color;
  voxel_ptr->weight = weight;
  return true;
}

__global__ void checkBlocksInTruncationBand(
    const VoxelBlock<TsdfVoxel>** block_device_ptrs,
    const float truncation_distance_m,
    bool* contains_truncation_band_device_ptr) {
  // A single thread in each block initializes the output to 0
  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
    contains_truncation_band_device_ptr[blockIdx.x] = 0;
  }
  __syncthreads();

  // Get the Voxel we'll check in this thread
  const TsdfVoxel voxel = block_device_ptrs[blockIdx.x]
                              ->voxels[threadIdx.z][threadIdx.y][threadIdx.x];

  // If this voxel in the truncation band, write the flag to say that the
  // block should be processed. NOTE(alexmillane): There will be collision on
  // write here. However, from my reading, all threads' writes will result in
  // a single write to global memory. Because we only write a single value (1)
  // it doesn't matter which thread "wins".
  if (std::abs(voxel.distance) <= truncation_distance_m) {
    contains_truncation_band_device_ptr[blockIdx.x] = true;
  }
}

std::vector<Index3D>
ProjectiveColorIntegrator::reduceBlocksToThoseInTruncationBand(
    const std::vector<Index3D>& block_indices, const TsdfLayer& tsdf_layer,
    const float truncation_distance_m) {
  // Check 1) Are the blocks allocated
  // - performed on the CPU because the hash-map is on the CPU
  std::vector<Index3D> block_indices_check_1;
  block_indices_check_1.reserve(block_indices.size());
  for (const Index3D& block_idx : block_indices) {
    if (tsdf_layer.isBlockAllocated(block_idx)) {
      block_indices_check_1.push_back(block_idx);
    }
  }

  if (block_indices_check_1.empty()) {
    return block_indices_check_1;
  }

  // Check 2) Does each of the blocks have a voxel within the truncation band
  // - performed on the GPU because the blocks are there
  // Get the blocks we need to check
  std::vector<const TsdfBlock*> block_ptrs =
      getBlockPtrsFromIndices(block_indices_check_1, tsdf_layer);

  const size_t num_blocks = block_ptrs.size();

  // Expand the buffers when needed
  if (num_blocks > truncation_band_block_ptrs_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserveAsync(new_size, *cuda_stream_);
    truncation_band_block_ptrs_device_.reserveAsync(new_size, *cuda_stream_);
    block_in_truncation_band_device_.reserveAsync(new_size, *cuda_stream_);
    block_in_truncation_band_host_.reserveAsync(new_size, *cuda_stream_);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_.copyFromAsync(block_ptrs, *cuda_stream_);
  truncation_band_block_ptrs_device_.copyFromAsync(
      truncation_band_block_ptrs_host_, *cuda_stream_);

  // Prepare output space
  block_in_truncation_band_device_.resizeAsync(num_blocks, *cuda_stream_);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  checkBlocksInTruncationBand<<<num_thread_blocks, kThreadsPerBlock, 0, *cuda_stream_>>>(
      truncation_band_block_ptrs_device_.data(),
      truncation_distance_m,
      block_in_truncation_band_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_.copyFromAsync(block_in_truncation_band_device_,
                                               *cuda_stream_);
  cuda_stream_->synchronize();

  // Filter the indices using the result
  std::vector<Index3D> block_indices_check_2;
  block_indices_check_2.reserve(block_indices_check_1.size());
  for (size_t i = 0; i < block_indices_check_1.size(); i++) {
    if (block_in_truncation_band_host_[i] == true) {
      block_indices_check_2.push_back(block_indices_check_1[i]);
    }
  }

  return block_indices_check_2;
}

}  // namespace nvblox
