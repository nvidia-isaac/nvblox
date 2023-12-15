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
#include <nvblox/integrators/projective_tsdf_integrator.h>

#include "nvblox/integrators/internal/cuda/impl/projective_integrator_impl.cuh"
#include "nvblox/integrators/internal/integrators_common.h"
#include "nvblox/integrators/weighting_function.h"

namespace nvblox {

struct UpdateTsdfVoxelFunctor {
  __host__ __device__ UpdateTsdfVoxelFunctor() = default;
  __host__ __device__ ~UpdateTsdfVoxelFunctor() = default;

  // Vector3f p_voxel_C, float depth, TsdfVoxel* voxel_ptr
  __device__ bool operator()(const float surface_depth_measured,
                             const float voxel_depth_m, TsdfVoxel* voxel_ptr) {
    // Get the distance between the voxel we're updating the surface.
    // Note that the distance is the projective distance, i.e. the distance
    // along the ray.
    const float voxel_to_surface_distance =
        surface_depth_measured - voxel_depth_m;

    // If we're behind the negative truncation distance, just continue.
    if (voxel_to_surface_distance < -truncation_distance_m_) {
      return false;
    }

    // Read CURRENT voxel values (from global GPU memory)
    const float voxel_distance_current = voxel_ptr->distance;
    const float voxel_weight_current = voxel_ptr->weight;

    // NOTE(alexmillane): We could try to use CUDA math functions to speed up
    // below
    // https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE

    // Get the weight of this observation from the sensor model.
    const float measurement_weight = weighting_function_(
        surface_depth_measured, voxel_depth_m, truncation_distance_m_);

    // Fuse
    float fused_distance = (voxel_to_surface_distance * measurement_weight +
                            voxel_distance_current * voxel_weight_current) /
                           (measurement_weight + voxel_weight_current);

    // Clip
    if (fused_distance > 0.0f) {
      fused_distance = fmin(truncation_distance_m_, fused_distance);
    } else {
      fused_distance = fmax(-truncation_distance_m_, fused_distance);
    }
    const float weight =
        fmin(measurement_weight + voxel_weight_current, max_weight_);

    // Write NEW voxel values (to global GPU memory)
    voxel_ptr->distance = fused_distance;
    voxel_ptr->weight = weight;
    return true;
  }

  float truncation_distance_m_ = 0.2f;
  float max_weight_ = ProjectiveTsdfIntegrator::kDefaultMaxWeight;

  WeightingFunction weighting_function_ = WeightingFunction(
      ProjectiveTsdfIntegrator::kDefaultWeightingFunctionType);
};

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator()
    : ProjectiveTsdfIntegrator(std::make_shared<CudaStreamOwning>()) {}

ProjectiveTsdfIntegrator::ProjectiveTsdfIntegrator(
    std::shared_ptr<CudaStream> cuda_stream)
    : ProjectiveIntegrator<TsdfVoxel>(cuda_stream) {
  update_functor_host_ptr_ =
      make_unified<UpdateTsdfVoxelFunctor>(MemoryType::kHost);
}

ProjectiveTsdfIntegrator::~ProjectiveTsdfIntegrator() {
  // NOTE(alexmillane): We can't default this in the header file because to the
  // unified_ptr to a forward declared type. The type has to be defined where
  // the destructor is.
}

unified_ptr<UpdateTsdfVoxelFunctor>
ProjectiveTsdfIntegrator::getTsdfUpdateFunctorOnDevice(float voxel_size) {
  // Set the update function params
  // NOTE(alex.millane): We do this with every frame integration to avoid
  // bug-prone logic for detecting when params have changed etc.
  update_functor_host_ptr_->max_weight_ = max_weight();
  update_functor_host_ptr_->truncation_distance_m_ =
      get_truncation_distance_m(voxel_size);
  update_functor_host_ptr_->weighting_function_ =
      WeightingFunction(weighting_function_type_);
  // Transfer to the device
  return update_functor_host_ptr_.cloneAsync(MemoryType::kDevice,
                                             *cuda_stream_);
}

void ProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Camera& camera,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  // Get the update functor on the device
  unified_ptr<UpdateTsdfVoxelFunctor> update_functor_device_ptr =
      getTsdfUpdateFunctorOnDevice(layer->voxel_size());
  // Integrate
  ProjectiveIntegrator<TsdfVoxel>::integrateFrame(
      depth_frame, T_L_C, camera, update_functor_device_ptr.get(), layer,
      updated_blocks);
}

void ProjectiveTsdfIntegrator::integrateFrame(
    const DepthImage& depth_frame, const Transform& T_L_C, const Lidar& lidar,
    TsdfLayer* layer, std::vector<Index3D>* updated_blocks) {
  // Get the update functor on the device
  unified_ptr<UpdateTsdfVoxelFunctor> update_functor_device_ptr =
      getTsdfUpdateFunctorOnDevice(layer->voxel_size());
  // Integrate
  ProjectiveIntegrator<TsdfVoxel>::integrateFrame(
      depth_frame, T_L_C, lidar, update_functor_device_ptr.get(), layer,
      updated_blocks);
}

float ProjectiveTsdfIntegrator::max_weight() const { return max_weight_; }

void ProjectiveTsdfIntegrator::max_weight(float max_weight) {
  CHECK_GT(max_weight, 0.0f);
  max_weight_ = max_weight;
}

WeightingFunctionType ProjectiveTsdfIntegrator::weighting_function_type()
    const {
  return weighting_function_type_;
}

void ProjectiveTsdfIntegrator::weighting_function_type(
    WeightingFunctionType weighting_function_type) {
  weighting_function_type_ = weighting_function_type;
}

float ProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m() const {
  return marked_unobserved_voxels_distance_m_;
}

void ProjectiveTsdfIntegrator::marked_unobserved_voxels_distance_m(
    float marked_unobserved_voxels_distance_m) {
  marked_unobserved_voxels_distance_m_ = marked_unobserved_voxels_distance_m;
}

float ProjectiveTsdfIntegrator::marked_unobserved_voxels_weight() const {
  return marked_unobserved_voxels_weight_;
}

void ProjectiveTsdfIntegrator::marked_unobserved_voxels_weight(
    float marked_unobserved_voxels_weight) {
  marked_unobserved_voxels_weight_ = marked_unobserved_voxels_weight;
}

std::string ProjectiveTsdfIntegrator::getIntegratorName() const {
  return "tsdf";
}

void ProjectiveTsdfIntegrator::markUnobservedFreeInsideRadius(
    const Vector3f& center, float radius, TsdfLayer* layer,
    std::vector<Index3D>* updated_blocks_ptr) {
  markUnobservedFreeInsideRadiusTemplate(center, radius, layer,
                                         updated_blocks_ptr);
}

parameters::ParameterTreeNode ProjectiveTsdfIntegrator::getParameterTree(
    const std::string& name_remap) const {
  using parameters::ParameterTreeNode;
  const std::string name =
      (name_remap.empty()) ? "projective_tsdf_integrator" : name_remap;
  // NOTE(alexmillane): Wrapping our weighting function to_string version in the
  // std::function for passing to the parameter tree node constructor because it
  // seems to have trouble with template deduction.
  std::function<std::string(const WeightingFunctionType&)>
      weighting_function_to_string =
          [](const WeightingFunctionType& w) { return to_string(w); };

  return ParameterTreeNode(
      name,
      {ParameterTreeNode("max_weight:", max_weight_),
       ParameterTreeNode("marked_unobserved_voxels_distance_m:",
                         marked_unobserved_voxels_distance_m_),
       ParameterTreeNode("marked_unobserved_voxels_weight:",
                         marked_unobserved_voxels_weight_),
       ParameterTreeNode("weighting_function_type:", weighting_function_type_,
                         weighting_function_to_string),
       ProjectiveIntegrator<TsdfVoxel>::getParameterTree()});
}

}  // namespace nvblox
