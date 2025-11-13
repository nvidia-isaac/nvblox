/*
 * Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "nvblox/core/color.h"
#include "nvblox/core/feature_array.h"
#include "nvblox/utils/cuda_kernel_utils.h"
#include "nvblox/utils/timing.h"
#include "nvblox_torch/check_utils.h"
#include "nvblox_torch/cuda_stream.h"
#include "nvblox_torch/py_mapper.h"
#include "nvblox_torch/sdf_query.cuh"

namespace pynvblox {

// declare nvblox variables here:
Mapper::Mapper(std::vector<double> voxel_size_m,
               std::vector<std::string> projective_layer_type,
               c10::intrusive_ptr<MapperParams> mapper_params) {
  // Initialize the mapper(s)
  mapper_params_ = mapper_params;
  voxel_size_m_ = voxel_size_m;
  projective_layer_type_ = projective_layer_type;
  const int num_mappers = voxel_size_m.size();
  CHECK_GT(num_mappers, 0);
  CHECK_EQ(num_mappers, static_cast<int>(projective_layer_type.size()));
  for (int i = 0; i < num_mappers; i++) {
    addMapper(voxel_size_m[i], projective_layer_type[i], *mapper_params);
  }

  // Cache the block sizes in a GPU buffer.
  std::vector<float> block_sizes_m;
  for (int i = 0; i < num_mappers; i++) {
    block_sizes_m.push_back(mappers_[i]->esdf_layer().block_size());
  }
  auto stream = getCurrentStream();
  block_sizes_m_gpu_.copyFromAsync(block_sizes_m, stream);
  stream.synchronize();
  CHECK_EQ(static_cast<int>(block_sizes_m_gpu_.size()), num_mappers);
}

void Mapper::addMapper(double voxel_size_m, std::string projective_layer_type,
                       const MapperParams& mapper_params) {
  nvblox::ProjectiveLayerType layer_type;
  if (projective_layer_type == "tsdf") {
    layer_type = nvblox::ProjectiveLayerType::kTsdf;
  } else if (projective_layer_type == "occupancy") {
    layer_type = nvblox::ProjectiveLayerType::kOccupancy;
  } else {
    LOG(FATAL) << "Invalid projective layer type: " << projective_layer_type;
  }

  // Create the mapper
  auto mapper = std::make_shared<nvblox::Mapper>(
      voxel_size_m, *mapper_params.get_block_memory_pool_params()->params_,
      layer_type);

  // Set parameters
  mapper->setMapperParams(*mapper_params.params_);

  // Add to mapper list
  mappers_.push_back(mapper);
}

long Mapper::getNumMappers() const { return mappers_.size(); }

std::shared_ptr<nvblox::Mapper> Mapper::getNvbloxMapper(long mapper_id) {
  CHECK_GE(mapper_id, 0);
  CHECK_LT(mapper_id, static_cast<long>(mappers_.size()));
  return mappers_[mapper_id];
}

c10::intrusive_ptr<MapperParams> Mapper::getMapperParams() {
  return mapper_params_;
}

void Mapper::integrateDepth(torch::Tensor depth_frame_t, torch::Tensor T_L_C_t,
                            torch::Tensor intrinsics_t,
                            std::optional<torch::Tensor> mask_frame_t,
                            long mapper_id) {
  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  ALL_ON_GPU_OR_RETURN(depth_frame_t, mask_frame_t);
  if (!checkSizes(T_L_C_t, {4, 4}) || !checkSizes(intrinsics_t, {3, 3})) {
    LOG(WARNING) << "Pose and intrinsics tensor sizes are not correct";
    return;
  }
  if (mask_frame_t.has_value() &&
      !checkImageDimensionsEqual(depth_frame_t, mask_frame_t.value())) {
    LOG(WARNING) << "Depth and mask frame sizes do not match";
    return;
  }

  auto mapper = mappers_[mapper_id];

  int height = depth_frame_t.sizes()[0];
  int width = depth_frame_t.sizes()[1];

  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);

  mapper->integrateDepth(
      masked_view_from_tensor<const float>(depth_frame_t, mask_frame_t), T_L_C,
      camera);
}

void Mapper::integrateColor(torch::Tensor color_frame_t, torch::Tensor T_L_C_t,
                            torch::Tensor intrinsics_t,
                            std::optional<torch::Tensor> mask_frame_t,
                            long mapper_id) {
  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  ALL_ON_GPU_OR_RETURN(color_frame_t, mask_frame_t);
  if (!checkSizes(T_L_C_t, {4, 4}) || !checkSizes(intrinsics_t, {3, 3})) {
    LOG(WARNING) << "Pose and intrinsics tensor sizes are not correct";
    return;
  }
  if (mask_frame_t.has_value() &&
      !checkImageDimensionsEqual(color_frame_t, mask_frame_t.value())) {
    LOG(WARNING) << "Color and mask frame sizes do not match";
    return;
  }

  auto mapper = mappers_[mapper_id];

  const int height = color_frame_t.sizes()[0];
  const int width = color_frame_t.sizes()[1];
  const int num_channels = color_frame_t.sizes()[2];
  CHECK_EQ(num_channels, nvblox::kRgbNumElements);
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);

  mapper->integrateColor(
      masked_view_from_tensor<const nvblox::Color>(color_frame_t, mask_frame_t),
      T_L_C, camera);
}

void Mapper::integrateFeatures(torch::Tensor feature_frame_t,
                               torch::Tensor T_L_C_t,
                               torch::Tensor intrinsics_t,
                               std::optional<torch::Tensor> mask_frame_t,
                               long mapper_id) {
  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  ALL_ON_GPU_OR_RETURN(feature_frame_t, mask_frame_t);
  if (!checkSizes(T_L_C_t, {4, 4}) || !checkSizes(intrinsics_t, {3, 3})) {
    LOG(WARNING) << "Pose and intrinsics tensor sizes are not correct";
    return;
  }
  if (mask_frame_t.has_value() &&
      !checkImageDimensionsEqual(feature_frame_t, mask_frame_t.value())) {
    LOG(WARNING) << "Feature and mask frame sizes do not match";
    return;
  }

  auto mapper = mappers_[mapper_id];

  int height = feature_frame_t.sizes()[0];
  int width = feature_frame_t.sizes()[1];
  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);

  mapper->integrateFeatures(masked_view_from_tensor<const nvblox::FeatureArray>(
                                feature_frame_t, mask_frame_t),
                            T_L_C, camera);
}

void Mapper::updateEsdf(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->updateEsdf();
  } else {
    for (auto& mapper : mappers_) {
      mapper->updateEsdf();
    }
  }
}

void Mapper::updateColorMesh(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->updateColorMesh();
  } else {
    for (auto& mapper : mappers_) {
      mapper->updateColorMesh();
    }
  }
}

void Mapper::updateFeatureMesh(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->updateFeatureMesh();
  } else {
    for (auto& mapper : mappers_) {
      mapper->updateFeatureMesh();
    }
  }
}

c10::intrusive_ptr<pynvblox::PyColorMesh> Mapper::getColorMesh(long mapper_id) {
  CHECK_LT(static_cast<size_t>(mapper_id), mappers_.size());
  CHECK_GE(mapper_id, 0);

  // Serialize all blocks in the layer
  constexpr float kUnlimitedBandwidth = -1.0F;
  auto mapper = mappers_[mapper_id];
  mapper->serializeSelectedLayers(
      nvblox::LayerType::kColorMesh, kUnlimitedBandwidth,
      nvblox::BlockExclusionParams(),
      mapper->color_mesh_layer().getAllBlockIndices());

  auto serialized_mesh = mapper->serializedColorMeshLayer();

  return c10::make_intrusive<pynvblox::PyColorMesh>(serialized_mesh);
}

c10::intrusive_ptr<pynvblox::PyFeatureMesh> Mapper::getFeatureMesh(
    long mapper_id) {
  CHECK_LT(static_cast<size_t>(mapper_id), mappers_.size());
  CHECK_GE(mapper_id, 0);

  // Serialize all blocks in the layer
  constexpr float kUnlimitedBandwidth = -1.0F;
  auto mapper = mappers_[mapper_id];
  mapper->serializeSelectedLayers(
      nvblox::LayerType::kFeatureMesh, kUnlimitedBandwidth,
      nvblox::BlockExclusionParams(),
      mapper->feature_mesh_layer().getAllBlockIndices());

  auto serialized_mesh = mapper->serializedFeatureMeshLayer();

  return c10::make_intrusive<pynvblox::PyFeatureMesh>(serialized_mesh);
}

void Mapper::fullUpdate(torch::Tensor depth_frame_t,
                        torch::Tensor color_frame_t, torch::Tensor T_L_C_t,
                        torch::Tensor intrinsics_t, long mapper_id) {
  auto mapper = mappers_[mapper_id];

  int height = depth_frame_t.sizes()[0];
  int width = depth_frame_t.sizes()[1];

  nvblox::Transform T_L_C = copy_transform_from_tensor(T_L_C_t);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics_t, height, width);

  mapper->integrateDepth(masked_view_from_tensor<const float>(depth_frame_t),
                         T_L_C, camera);
  mapper->integrateColor(
      masked_view_from_tensor<const nvblox::Color>(color_frame_t), T_L_C,
      camera);

  mapper->updateEsdf();
  mapper->updateColorMesh();
  mapper->updateFeatureMesh();
}

void Mapper::decayTsdf(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->decayTsdfAllVoxels();

  } else {
    for (auto& mapper : mappers_) {
      mapper->decayTsdfAllVoxels();
    }
  }
}

void Mapper::decayOccupancy(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->decayOccupancyAllVoxels();

  } else {
    for (auto& mapper : mappers_) {
      mapper->decayOccupancyAllVoxels();
    }
  }
}

void Mapper::clear(long mapper_id) {
  if (mapper_id >= 0) {
    mappers_[mapper_id]->occupancy_layer().clear();
    mappers_[mapper_id]->tsdf_layer().clear();
    mappers_[mapper_id]->esdf_layer().clear();
    mappers_[mapper_id]->color_layer().clear();
    mappers_[mapper_id]->color_mesh_layer().clear();
    mappers_[mapper_id]->feature_mesh_layer().clear();
    mappers_[mapper_id]->feature_layer().clear();
  } else {
    for (auto& mapper : mappers_) {
      mapper->occupancy_layer().clear();
      mapper->tsdf_layer().clear();
      mapper->esdf_layer().clear();
      mapper->color_layer().clear();
      mapper->color_mesh_layer().clear();
      mapper->feature_mesh_layer().clear();
      mapper->feature_layer().clear();
    }
  }
}

c10::intrusive_ptr<PyTsdfLayer> Mapper::tsdf_layer(const long mapper_id) {
  return get_layer<PyTsdfLayer>(mapper_id, "tsdf",
                                nvblox::ProjectiveLayerType::kTsdf);
}

c10::intrusive_ptr<PyColorLayer> Mapper::color_layer(const long mapper_id) {
  return get_layer<PyColorLayer>(mapper_id, "color");
}

c10::intrusive_ptr<PyFeatureLayer> Mapper::feature_layer(const long mapper_id) {
  return get_layer<PyFeatureLayer>(mapper_id, "feature");
}

template <typename PyLayerType>
c10::intrusive_ptr<PyLayerType> Mapper::get_layer(
    const long mapper_id, const std::string& name,
    nvblox::ProjectiveLayerType required_projective_layer_type) {
  CHECK_GE(mapper_id, 0);
  CHECK_LT(static_cast<size_t>(mapper_id), mappers_.size());
  auto mapper = mappers_[mapper_id];

  if (required_projective_layer_type != nvblox::ProjectiveLayerType::kNone &&
      mapper->projective_layer_type() != required_projective_layer_type) {
    LOG(ERROR) << "Requested a " << name
               << " layer from a mapper not configured for "
               << nvblox::toString(required_projective_layer_type)
               << " mapping. Returning an empty layer.";
    return c10::make_intrusive<PyLayerType>(voxel_size_m_[mapper_id]);
  }

  auto ptr = c10::make_intrusive<PyLayerType>(
      mapper->layers().getSharedPtr<typename PyLayerType::NativeLayerType>());

  if (ptr == nullptr) {
    LOG(ERROR) << "Requested a " << name
               << " layer which does not exist in mapper with id " << mapper_id
               << ". Returning an empty layer.";
  }
  return ptr;
}

torch::Tensor Mapper::renderDepthImage(torch::Tensor camera_pose,
                                       torch::Tensor intrinsics,
                                       int64_t img_height, int64_t img_width,
                                       double max_ray_length, int64_t max_steps,
                                       long mapper_id) {
  auto mapper = mappers_[mapper_id];

  // TODO: This 4.0 is the default truncation distance in
  // projective_integrator_base.h This should be made a global constant and
  // somehow set accordingly.
  double truncation_distance_m = voxel_size_m_[mapper_id] * 4.0;

  nvblox::Transform T_S_C = copy_transform_from_tensor(camera_pose);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics, img_height, img_width);

  nvblox::SphereTracer sphere_tracer_gpu;
  sphere_tracer_gpu.maximum_ray_length_m(max_ray_length);
  sphere_tracer_gpu.maximum_steps(max_steps);

  nvblox::TsdfLayer& layer = mapper->tsdf_layer();
  torch::DeviceType device =
      torch::kCUDA;  // Currently SphereTracer only supports GPU)

  torch::Tensor depth_image_t =
      init_depth_image_tensor(img_height, img_width, device);
  nvblox::DepthImageView depth_image_view =
      view_from_tensor<float>(depth_image_t);
  sphere_tracer_gpu.renderImageOnGPU(camera, T_S_C, layer,
                                     truncation_distance_m, &depth_image_view,
                                     nvblox::MemoryType::kDevice);

  return depth_image_t;
}

std::vector<torch::Tensor> Mapper::renderDepthAndColorImage(
    torch::Tensor camera_pose, torch::Tensor intrinsics, int64_t img_height,
    int64_t img_width, double max_ray_length, int64_t max_steps,
    long mapper_id) {
  auto mapper = mappers_[mapper_id];
  // TODO: This 4.0 is the default truncation distance in
  // projective_integrator_base.h This should be made a global constant and
  // somehow set accordingly.
  double truncation_distance_m = voxel_size_m_[mapper_id] * 4.0;

  nvblox::Transform T_S_C = copy_transform_from_tensor(camera_pose);
  nvblox::Camera camera =
      camera_from_intrinsics_tensor(intrinsics, img_height, img_width);

  nvblox::SphereTracer sphere_tracer_gpu;
  sphere_tracer_gpu.maximum_ray_length_m(max_ray_length);
  sphere_tracer_gpu.maximum_steps(max_steps);

  nvblox::TsdfLayer& tsdf_layer = mapper->tsdf_layer();
  nvblox::ColorLayer& color_layer = mapper->color_layer();

  torch::DeviceType device =
      torch::kCUDA;  // Currently SphereTracer only supports GPU

  torch::Tensor depth_image_t =
      init_depth_image_tensor(img_height, img_width, device);
  nvblox::DepthImageView depth_image_view =
      view_from_tensor<float>(depth_image_t);
  torch::Tensor color_image_t =
      init_color_image_tensor(img_height, img_width, device);
  nvblox::ColorImageView color_image_view =
      view_from_tensor<nvblox::Color>(color_image_t);

  sphere_tracer_gpu.renderRgbdImageOnGPU(
      camera, T_S_C, tsdf_layer, color_layer, truncation_distance_m,
      &depth_image_view, &color_image_view, nvblox::MemoryType::kDevice);

  return {depth_image_t, color_image_t};
}

bool Mapper::outputColorMeshPly(std::string mesh_output_path, long mapper_id) {
  auto mapper = mappers_[mapper_id];
  return nvblox::io::outputColorMeshLayerToPly(mapper->color_mesh_layer(),
                                               mesh_output_path.c_str());
}

bool Mapper::outputBloxMap(std::string blox_output_path, long mapper_id) {
  auto mapper = mappers_[mapper_id];
  const bool result = mapper->saveLayerCake(blox_output_path);
  return result;
}

template <typename BlockType>
void Mapper::transferGPUHashesAsync(
    nvblox::host_vector<nvblox::Index3DDeviceHashMapType<BlockType>>*
        hash_transfer_buffer_host,
    nvblox::device_vector<nvblox::Index3DDeviceHashMapType<BlockType>>*
        hash_transfer_buffer_device,
    const nvblox::CudaStream& stream) {
  // Loop through all mappers and copy the hashes to the host and device
  // buffers.
  using LayerType = nvblox::VoxelBlockLayer<typename BlockType::VoxelType>;
  const int num_mappers = static_cast<int>(mappers_.size());
  hash_transfer_buffer_host->clearNoDeallocate();
  for (int i = 0; i < num_mappers; i++) {
    nvblox::GPULayerView<BlockType>& gpu_layer_view =
        mappers_[i]->layers().get<LayerType>().getGpuLayerView(stream);
    hash_transfer_buffer_host->push_back(gpu_layer_view.getHash().impl_);
  }
  hash_transfer_buffer_device->clearNoDeallocate();
  hash_transfer_buffer_device->copyFromAsync(*hash_transfer_buffer_host,
                                             stream);
  CHECK_EQ(static_cast<int>(hash_transfer_buffer_device->size()), num_mappers);
  CHECK_EQ(static_cast<int>(hash_transfer_buffer_host->size()), num_mappers);
}

torch::Tensor Mapper::queryMultiEsdf(torch::Tensor output_tensor,
                                     const torch::Tensor query_sphere) {
  const int64_t num_queries = query_sphere.sizes()[0];
  // Check inputs
  if (!checkAllOnGPU(output_tensor, query_sphere)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    std::cout << "Inputs need to be accessible on the GPU." << std::endl;
    return torch::empty({0});
  }
  if (!checkSizes(query_sphere, {static_cast<int>(num_queries), 4})) {
    LOG(ERROR) << "Inputs do not have the required sizes";
    return torch::empty({0});
  }

  if (!checkSizes(output_tensor, {static_cast<int>(num_queries), 4}) &&
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 1})) {
    LOG(ERROR) << "Output has to be either (Nx1) or (Nx4)";
    return torch::empty({0});
  }

  CHECK_GT(mappers_.size(), static_cast<size_t>(0));

  // Do we need gradients? That's governed by the output tensor size
  const bool extract_gradients = output_tensor.sizes()[1] == 4;

  // Stream
  auto stream = getCurrentStream();

  // GPU hash transfer
  transferGPUHashesAsync<nvblox::EsdfBlock>(&esdf_hash_transfer_buffer_host_,
                                            &esdf_hash_transfer_buffer_device_,
                                            stream);

  // Call the kernel.
  const int num_mappers = static_cast<int>(mappers_.size());
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);

  pynvblox::sdf::
      queryESDFMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
          num_mappers, num_queries, extract_gradients,
          esdf_hash_transfer_buffer_device_.data(), block_sizes_m_gpu_.data(),
          query_sphere.data_ptr<float>(), output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

torch::Tensor Mapper::queryMultiOccupancy(torch::Tensor output_tensor,
                                          const torch::Tensor query_positions) {
  const int64_t num_queries = query_positions.sizes()[0];

  // Check inputs
  if (!checkAllOnGPU(output_tensor, query_positions)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    std::cout << "Inputs need to be accessible on the GPU." << std::endl;
    return torch::empty({0});
  }
  if (!checkSizes(query_positions, {static_cast<int>(num_queries), 3}) ||
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 1})) {
    LOG(ERROR) << "Inputs do not have the required sizes";
    return torch::empty({0});
  }
  CHECK_GT(mappers_.size(), static_cast<size_t>(0));

  // Transfer hashes to GPU.
  auto stream = getCurrentStream();
  transferGPUHashesAsync<nvblox::OccupancyBlock>(
      &occupancy_hash_transfer_buffer_host_,
      &occupancy_hash_transfer_buffer_device_, stream);

  // Call a kernel.
  const int num_mappers = mappers_.size();
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  float* out_log_odds = output_tensor.data_ptr<float>();
  pynvblox::sdf::
      queryOccupancyMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
          num_mappers, num_queries,
          occupancy_hash_transfer_buffer_device_.data(),
          block_sizes_m_gpu_.data(), query_positions.data_ptr<float>(),
          out_log_odds);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

torch::Tensor Mapper::queryEsdf(torch::Tensor output_tensor,
                                const torch::Tensor query_sphere,
                                long mapper_id) {
  const int64_t num_queries = query_sphere.sizes()[0];
  // Input checks.
  if (!checkAllOnGPU(output_tensor, query_sphere)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }
  if (!checkSizes(query_sphere, {static_cast<int>(num_queries), 4})) {
    LOG(ERROR) << "Inputs do not have the required sizes";
    return torch::empty({0});
  }
  if (!checkSizes(output_tensor, {static_cast<int>(num_queries), 4}) &&
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 1})) {
    LOG(ERROR) << "Output has to be either (Nx1) or (Nx4)";
    return torch::empty({0});
  }

  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  CHECK_GE(mapper_id, 0);

  // Do we need gradients? That's governed by the output tensor size
  const bool extract_gradients = output_tensor.sizes()[1] == 4;

  // Get the mapper to query
  auto mapper = mappers_[mapper_id];

  auto stream = getCurrentStream();

  // GPU hash transfer
  nvblox::GPULayerView<nvblox::EsdfBlock>& gpu_layer_view =
      mapper->esdf_layer().getGpuLayerView(stream);

  // Call a kernel.
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  pynvblox::sdf::queryESDFKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_queries, extract_gradients, gpu_layer_view.getHash().impl_,
      mapper->esdf_layer().block_size(), query_sphere.data_ptr<float>(),
      output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

torch::Tensor Mapper::queryFeatures(torch::Tensor output_tensor,
                                    const torch::Tensor query_positions,
                                    long mapper_id) {
  const int64_t num_queries = query_positions.sizes()[0];

  // Input checks.
  if (!checkAllOnGPU(output_tensor, query_positions)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }

  constexpr int kPositionNumElements = 3;
  if (!checkSizes(query_positions,
                  {static_cast<int>(num_queries), kPositionNumElements})) {
    LOG(ERROR) << "Query positions do not have the required size.";
    return torch::empty({0});
  }

  constexpr int kOutputNumElements =
      nvblox::FeatureArray::size() + 1;  // +1 for the weight
  if (!checkSizes(output_tensor,
                  {static_cast<int>(num_queries), kOutputNumElements})) {
    LOG(ERROR) << "Output features tensor does not have the required size.";
    return torch::empty({0});
  }

  if (!checkElementSize<float>(query_positions)) {
    LOG(ERROR) << "Input query points tensor has to be of type float32";
    return torch::empty({0});
  }

  if (!checkElementSize<nvblox::FeatureVoxel::ArrayType::value_type>(
          output_tensor)) {
    LOG(ERROR) << "Output feature tensor has wrong size.";
    return torch::empty({0});
  }

  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  CHECK_GE(mapper_id, 0);

  // Get the mapper to query
  auto mapper = mappers_[mapper_id];

  auto stream = getCurrentStream();

  // GPU hash transfer
  nvblox::GPULayerView<nvblox::FeatureBlock>& gpu_layer_view =
      mapper->feature_layer().getGpuLayerView(stream);

  // Call a kernel.
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);

  pynvblox::sdf::queryFeatureKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_queries, gpu_layer_view.getHash().impl_,
      mapper->feature_layer().block_size(), query_positions.data_ptr<float>(),
      output_tensor.data_ptr<at::Half>());

  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

torch::Tensor Mapper::queryTsdf(torch::Tensor output_tensor,
                                const torch::Tensor query_positions,
                                long mapper_id) {
  const int64_t num_queries = query_positions.sizes()[0];

  // Input checks.
  if (!checkAllOnGPU(output_tensor, query_positions)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }
  if (!checkSizes(query_positions, {static_cast<int>(num_queries), 3}) ||
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 2})) {
    LOG(ERROR) << "Inputs do not have the required sizes";
    return torch::empty({0});
  }
  CHECK_LT(mapper_id, static_cast<int>(mappers_.size()));
  CHECK_GE(mapper_id, 0);

  // Get the mapper to query
  auto mapper = mappers_[mapper_id];

  auto stream = getCurrentStream();

  // GPU hash transfer
  nvblox::GPULayerView<nvblox::TsdfBlock>& gpu_layer_view =
      mapper->tsdf_layer().getGpuLayerView(stream);

  // Call a kernel.
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  pynvblox::sdf::queryTSDFKernel<<<num_blocks, kNumThreads, 0, stream>>>(
      num_queries, gpu_layer_view.getHash().impl_,
      mapper->tsdf_layer().block_size(), query_positions.data_ptr<float>(),
      output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

torch::Tensor Mapper::queryMultiTsdf(torch::Tensor output_tensor,
                                     const torch::Tensor query_positions) {
  const int64_t num_queries = query_positions.sizes()[0];

  // Input checks.
  if (!checkAllOnGPU(output_tensor, query_positions)) {
    LOG(ERROR) << "Inputs need to be accessible on the GPU.";
    return torch::empty({0});
  }
  if (!checkSizes(query_positions, {static_cast<int>(num_queries), 3}) ||
      !checkSizes(output_tensor, {static_cast<int>(num_queries), 2})) {
    LOG(ERROR) << "Inputs do not have the required sizes";
    return torch::empty({0});
  }

  // Transfer hashes to GPU.
  auto stream = getCurrentStream();
  transferGPUHashesAsync<nvblox::TsdfBlock>(&tsdf_hash_transfer_buffer_host_,
                                            &tsdf_hash_transfer_buffer_device_,
                                            stream);

  // Call kernel.
  const int num_mappers = mappers_.size();
  constexpr int kNumThreads = 128;
  int num_blocks = nvblox::divideRoundUp(num_queries, kNumThreads);
  pynvblox::sdf::
      queryTSDFMultiMapperKernel<<<num_blocks, kNumThreads, 0, stream>>>(
          num_mappers, num_queries, tsdf_hash_transfer_buffer_device_.data(),
          block_sizes_m_gpu_.data(), query_positions.data_ptr<float>(),
          output_tensor.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  stream.synchronize();

  return output_tensor;
}

void Mapper::loadFromFile(std::string file_path, long mapper_id) {
  // TODO: How to load?
  // mapper_.reset(new RgbdMapper(file_path, MemoryType::kDevice));
  mappers_[mapper_id]->loadMap(file_path.c_str());
}

std::string Mapper::printTiming() const {
  return nvblox::timing::Timing::Print();
}

}  // namespace pynvblox
