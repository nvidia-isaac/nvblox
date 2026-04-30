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

#include "nvblox/core/array.h"
#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/cuda/projective_integrators_common.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/projective_appearance_integrator.h"
#include "nvblox/integrators/projective_integrator_params.h"
#include "nvblox/interpolation/interpolation_2d.h"
#include "nvblox/utils/timing.h"

namespace nvblox {

// Return the array used for integration given a voxel. Needs an overload for
// all supported voxel types
__host__ __device__ Color& getArrayFromAppearanceVoxel(ColorVoxel& voxel) {
  return voxel.color;
}
__host__ __device__ FeatureArray& getArrayFromAppearanceVoxel(
    FeatureVoxel& voxel) {
  return voxel.feature;
}

// Set the integrated array of a voxel. Needs an overload for all supported
// voxel types
__host__ __device__ void setArrayFromAppearanceVoxel(const Color& color,
                                                     ColorVoxel* voxel) {
  voxel->color = color;
}
__host__ __device__ void setArrayFromAppearanceVoxel(
    const FeatureArray& feature, FeatureVoxel* voxel) {
  voxel->feature = feature;
}

template <class LayerType>
ProjectiveAppearanceIntegrator<LayerType>::ProjectiveAppearanceIntegrator()
    : ProjectiveAppearanceIntegrator(std::make_shared<CudaStreamOwning>()) {}

template <class LayerType>
ProjectiveAppearanceIntegrator<LayerType>::ProjectiveAppearanceIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : ProjectiveIntegrator<VoxelType>(cuda_stream),
      update_functor_host_ptr_(
          make_unified<UpdateAppearanceVoxelFunctor<VoxelType>>(
              MemoryType::kHost)),
      sphere_tracer_(cuda_stream) {
  sphere_tracer_.maximum_ray_length_m(this->max_integration_distance_m_);
}

// NOTE(dtingdahl): We can't default this in the header file because to the
// unified_ptr to a forward declared type. The type has to be defined where
// the destructor is.
template <class LayerType>
ProjectiveAppearanceIntegrator<LayerType>::~ProjectiveAppearanceIntegrator() =
    default;

template <class LayerType>
void ProjectiveAppearanceIntegrator<LayerType>::integrateFrame(
    const MaskedImageType& image,
    std::optional<MaskedDepthImageConstView> depth_image,
    const Transform& T_L_C, const Camera& camera, const TsdfLayer& tsdf_layer,
    LayerType* layer, std::vector<Index3D>* updated_blocks) {
  timing::Timer timer(getIntegratorName() + "/integrate");
  CHECK_NOTNULL(layer);
  CHECK_EQ(tsdf_layer.block_size(), layer->block_size());

  const float truncation_distance_m =
      this->truncation_distance_vox_ * tsdf_layer.voxel_size();

  // Get blocks in view. When a depth image is available we can use raycasting
  // for tighter block selection; otherwise fall back to frustum projection.
  timing::Timer blocks_in_view_timer(getIntegratorName() +
                                     "/integrate/get_blocks_in_view");
  std::vector<Index3D> block_indices;
  if (depth_image.has_value()) {
    block_indices = view_calculator_.getBlocksInImageViewRaycast(
        *depth_image, T_L_C, camera, layer->block_size(), truncation_distance_m,
        this->max_integration_distance_m_);
  } else {
    block_indices = view_calculator_.getBlocksInImageViewProjection(
        T_L_C, camera, layer->block_size(),
        this->max_integration_distance_m_ + truncation_distance_m);
  }
  blocks_in_view_timer.Stop();

  if (block_indices.empty()) {
    return;
  }

  // Reduce to blocks allocated in the TSDF with at least one voxel in the
  // truncation band. We only paint existing geometry, not freespace.
  timing::Timer blocks_in_band_timer(getIntegratorName() +
                                     "/integrate/reduce_to_blocks_in_band");
  block_indices = reduceBlocksToThoseInTruncationBand(block_indices, tsdf_layer,
                                                      truncation_distance_m);
  blocks_in_band_timer.Stop();

  if (block_indices.empty()) {
    return;
  }

  // Allocate blocks (CPU)
  // We allocate blocks where
  // - there are allocated TSDF blocks, AND
  // - these blocks are within the truncation band
  timing::Timer allocate_blocks_timer(getIntegratorName() +
                                      "/integrate/allocate_blocks");
  allocateBlocksWhereRequired(block_indices, layer, *this->cuda_stream_);
  allocate_blocks_timer.Stop();

  // When no depth image is provided, generate one via sphere tracing.
  if (!depth_image.has_value()) {
    const SphereTracer::SubsampledImageSize image_size =
        sphere_tracer_.getSubsampledImageSize(
            camera, sphere_tracing_ray_subsampling_factor_);
    DepthImage* synthetic_depth_image = synthetic_depth_images_.get(
        image_size.rows, image_size.cols, MemoryType::kDevice);

    timing::Timer sphere_trace_timer(getIntegratorName() +
                                     "/integrate/sphere_trace");
    sphere_tracer_.renderImageOnGPU(
        camera, T_L_C, tsdf_layer, truncation_distance_m, synthetic_depth_image,
        MemoryType::kDevice, sphere_tracing_ray_subsampling_factor_);
    sphere_trace_timer.Stop();

    depth_image.emplace(*synthetic_depth_image, kMaskActiveEverywhere);
  }

  timing::Timer transfer_blocks_timer(getIntegratorName() +
                                      "/integrate/transfer_blocks");
  transferBlockPointersToDeviceAsync<BlockType>(
      block_indices, layer, &this->block_ptrs_host_, &this->block_ptrs_device_,
      *this->cuda_stream_);
  transferBlockIndicesToDeviceAsync(block_indices, &this->block_indices_host_,
                                    &this->block_indices_device_,
                                    *this->cuda_stream_);

  // We need the inverse transform in the kernel
  const Transform T_C_L = T_L_C.inverse();

  // Move the functor to the GPU
  unified_ptr<UpdateAppearanceVoxelFunctor<VoxelType>> update_functor_device =
      getAppearanceUpdateFunctorOnDevice(tsdf_layer.voxel_size());
  transfer_blocks_timer.Stop();

  // Calling the GPU to do the updates
  timing::Timer update_blocks_timer(getIntegratorName() +
                                    "/integrate/update_blocks");
  this->integrateBlocks(*depth_image, image, T_C_L, camera,
                        update_functor_device.get(), layer);
  update_blocks_timer.Stop();

  if (updated_blocks != nullptr) {
    *updated_blocks = block_indices;
  }
}

template <class LayerType>
void ProjectiveAppearanceIntegrator<LayerType>::
    sphere_tracing_ray_subsampling_factor(
        int sphere_tracing_ray_subsampling_factor) {
  CHECK_GT(sphere_tracing_ray_subsampling_factor, 0);
  sphere_tracing_ray_subsampling_factor_ =
      sphere_tracing_ray_subsampling_factor;
}

template <class LayerType>
int ProjectiveAppearanceIntegrator<
    LayerType>::sphere_tracing_ray_subsampling_factor() const {
  return sphere_tracing_ray_subsampling_factor_;
}

template <class LayerType>
float ProjectiveAppearanceIntegrator<LayerType>::max_weight() const {
  return max_weight_;
}

template <class LayerType>
void ProjectiveAppearanceIntegrator<LayerType>::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

template <class LayerType>
float ProjectiveAppearanceIntegrator<LayerType>::measurement_weight() const {
  return measurement_weight_;
}

template <class LayerType>
void ProjectiveAppearanceIntegrator<LayerType>::measurement_weight(
    float measurement_weight) {
  CHECK_GT(measurement_weight, 0.0f);
  CHECK_LE(measurement_weight, 1.0f);
  measurement_weight_ = measurement_weight;
}

template <class LayerType>
float ProjectiveAppearanceIntegrator<LayerType>::get_truncation_distance_m(
    float voxel_size) const {
  return this->truncation_distance_vox_ * voxel_size;
}

template <class LayerType>
WeightingFunctionType
ProjectiveAppearanceIntegrator<LayerType>::weighting_function_type() const {
  return weighting_function_type_;
}

template <class LayerType>
void ProjectiveAppearanceIntegrator<LayerType>::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

template <class LayerType>
const ViewCalculator&
ProjectiveAppearanceIntegrator<LayerType>::view_calculator() const {
  return view_calculator_;
}

/// Returns the object used to calculate the blocks in camera views.
template <class LayerType>
ViewCalculator& ProjectiveAppearanceIntegrator<LayerType>::view_calculator() {
  return view_calculator_;
}

template <class LayerType>
parameters::ParameterTreeNode
ProjectiveAppearanceIntegrator<LayerType>::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "projective_appearance_integrator" : name_remap;
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
                ParameterTreeNode("measurement_weight:", measurement_weight_),
                ParameterTreeNode(
                    "weighting_function_type:", weighting_function_type_,
                    weighting_function_to_string),
                ProjectiveIntegrator<VoxelType>::getParameterTree(),
                view_calculator_.getParameterTree(),
            });
}

template <typename FloatType>
__device__ inline FloatType weightedSum(const FloatType first_value,
                                        const FloatType first_weight,
                                        const FloatType second_value,
                                        const FloatType second_weight) {
  static_assert(isFloatType<FloatType>(),
                "Only floating point types supported");
  return first_value * first_weight + second_value * second_weight;
}

__device__ inline uint8_t weightedSum(const uint8_t first_value,
                                      const float first_weight,
                                      const uint8_t second_value,
                                      const float second_weight) {
  return static_cast<uint8_t>(
      std::round(static_cast<float>(first_value) * first_weight +
                 static_cast<float>(second_value) * second_weight));
}

template <class ArrayType>
__device__ inline void blendTwoArrays(const ArrayType& first_array,
                                      float first_weight,
                                      const ArrayType& second_array,
                                      float second_weight,
                                      ArrayType* new_array) {
  float total_weight = first_weight + second_weight;

  first_weight /= total_weight;
  second_weight /= total_weight;

  NVBLOX_DCHECK(first_weight >= 0.F, "Weights must be positive");
  NVBLOX_DCHECK(second_weight >= 0.F, "Weights must be positive");
  NVBLOX_DCHECK(new_array != nullptr, "");

  for (size_t i = 0; i < first_array.size(); ++i) {
    (*new_array)[i] = weightedSum(first_array[i], __float2half(first_weight),
                                  second_array[i], __float2half(second_weight));
  }
}

template <class VoxelType>
struct UpdateAppearanceVoxelFunctor {
  __host__ __device__ UpdateAppearanceVoxelFunctor() = default;
  __host__ __device__ ~UpdateAppearanceVoxelFunctor() = default;

  using ArrayType = typename VoxelType::ArrayType;

  __device__ bool operator()(
      const float measured_depth_m, const float voxel_depth_m,
      const bool is_active,
      const std::optional<typename VoxelType::ArrayType>& appearance_measured,
      VoxelType* voxel_ptr) {
    NVBLOX_CHECK(appearance_measured.has_value(), "Need measurement");

    // If the mask is inactive, we skip this measurement
    if (!is_active) {
      return false;
    }

    // Read CURRENT voxel values (from global GPU memory)
    const ArrayType voxel_appearance_current =
        getArrayFromAppearanceVoxel(*voxel_ptr);
    const float voxel_weight_current = voxel_ptr->weight;

    // Fuse measurement with current estimate
    ArrayType fused_appearance;
    if (__half2float(voxel_ptr->weight) == 0.f) {
      // If this is the first measurement, we simply copy the measurement
      setArrayFromAppearanceVoxel(appearance_measured.value(), voxel_ptr);
    } else {
      // Exponential filter
      blendTwoArrays(voxel_appearance_current, (1.0f - measurement_weight_),
                     appearance_measured.value(), measurement_weight_,
                     &fused_appearance);
      // Write NEW voxel values (to global GPU memory)
      setArrayFromAppearanceVoxel(fused_appearance, voxel_ptr);
    }

    voxel_ptr->weight =
        fmin(measurement_weight_ + voxel_weight_current, max_weight_);

    return true;
  }
  WeightingFunction weighting_function_ =
      kProjectiveIntegratorWeightingModeParamDesc.default_value;
  float truncation_distance_m_ = 0.2f;
  float max_weight_ = kProjectiveIntegratorMaxWeightParamDesc.default_value;
  float measurement_weight_ =
      kProjectiveAppearanceIntegratorMeasurementWeightParamDesc.default_value;
};

template <class LayerType>
unified_ptr<UpdateAppearanceVoxelFunctor<typename LayerType::VoxelType>>
ProjectiveAppearanceIntegrator<LayerType>::getAppearanceUpdateFunctorOnDevice(
    float voxel_size) {
  // Set the update function params
  // NOTE(alex.millane): We do this with every frame integration to avoid
  // bug-prone logic for detecting when params have changed etc.
  CHECK(update_functor_host_ptr_ != nullptr);
  update_functor_host_ptr_->max_weight_ = max_weight();
  update_functor_host_ptr_->measurement_weight_ = measurement_weight();
  update_functor_host_ptr_->truncation_distance_m_ =
      get_truncation_distance_m(voxel_size);
  update_functor_host_ptr_->weighting_function_ =
      WeightingFunction(weighting_function_type_);
  // Transfer to the device
  return update_functor_host_ptr_.cloneAsync(MemoryType::kDevice,
                                             *this->cuda_stream_);
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
  // Note that "strictly less" is needed here to avoid picking up all the voxels
  // with a truncated distance.
  if (voxel.weight > 0.F && std::abs(voxel.distance) < truncation_distance_m) {
    contains_truncation_band_device_ptr[blockIdx.x] = true;
  }
}

template <class LayerType>
std::vector<Index3D>
ProjectiveAppearanceIntegrator<LayerType>::reduceBlocksToThoseInTruncationBand(
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
  if (num_blocks > this->truncation_band_block_ptrs_device_.capacity()) {
    constexpr float kBufferExpansionFactor = 1.5f;
    const int new_size = static_cast<int>(kBufferExpansionFactor * num_blocks);
    truncation_band_block_ptrs_host_.reserveAsync(new_size,
                                                  *this->cuda_stream_);
    truncation_band_block_ptrs_device_.reserveAsync(new_size,
                                                    *this->cuda_stream_);
    block_in_truncation_band_device_.reserveAsync(new_size,
                                                  *this->cuda_stream_);
    block_in_truncation_band_host_.reserveAsync(new_size, *this->cuda_stream_);
  }

  // Host -> Device
  truncation_band_block_ptrs_host_.copyFromAsync(block_ptrs,
                                                 *this->cuda_stream_);
  truncation_band_block_ptrs_device_.copyFromAsync(
      truncation_band_block_ptrs_host_, *this->cuda_stream_);

  // Prepare output space
  block_in_truncation_band_device_.resizeAsync(num_blocks, *this->cuda_stream_);

  // Do the check on GPU
  // Kernel call - One ThreadBlock launched per VoxelBlock
  constexpr int kVoxelsPerSide = VoxelBlock<bool>::kVoxelsPerSide;
  const dim3 kThreadsPerBlock(kVoxelsPerSide, kVoxelsPerSide, kVoxelsPerSide);
  const int num_thread_blocks = num_blocks;
  // clang-format off
  checkBlocksInTruncationBand<<<num_thread_blocks, kThreadsPerBlock, 0, *this->cuda_stream_>>>(
      truncation_band_block_ptrs_device_.data(),
      truncation_distance_m,
      block_in_truncation_band_device_.data());
  // clang-format on
  checkCudaErrors(cudaPeekAtLastError());

  // Copy results back
  block_in_truncation_band_host_.copyFromAsync(block_in_truncation_band_device_,
                                               *this->cuda_stream_);
  this->cuda_stream_->synchronize();

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

// Specializations
template <>
std::string ProjectiveColorIntegrator::getIntegratorName() const {
  return "color";
}
template <>
std::string ProjectiveFeatureIntegrator::getIntegratorName() const {
  return "feature";
}

// Instantiate the integrators
template class ProjectiveAppearanceIntegrator<ColorLayer>;
template class ProjectiveAppearanceIntegrator<FeatureLayer>;

}  // namespace nvblox
