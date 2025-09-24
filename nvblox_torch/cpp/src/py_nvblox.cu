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

#include "nvblox_torch/py_mapper.h"

#include <torch/script.h>

#include <ATen/ATen.h>
#include <torch/custom_class.h>

#include "nvblox_torch/py_constants.h"
#include "nvblox_torch/py_layer.h"
#include "nvblox_torch/py_mapper_params.h"
#include "nvblox_torch/py_mesh.h"
#include "nvblox_torch/py_rendering.h"
#include "nvblox_torch/py_scene.h"

namespace pynvblox {

/// Function for declaring a voxel block layer with a given name
template <typename LayerType>
void defineVoxelBlockLayerClass(torch::Library& m, const std::string& name) {
  m.class_<LayerType>(name)
      .def(torch::init<double>())
      .def("clear", &LayerType::clear)
      .def("allocate_block_at_index", &LayerType::allocateBlockAtIndex)
      .def("is_block_allocated", &LayerType::isBlockAllocated)
      .def("get_block_at_index", &LayerType::getBlockAtIndex)
      .def("get_all_block_indices", &LayerType::getAllBlockIndices)
      .def("get_all_blocks", &LayerType::getAllBlocks)
      .def("num_blocks", &LayerType::numBlocks)
      .def("voxel_size", &LayerType::voxel_size)
      .def("num_allocated_bytes", &LayerType::numAllocatedBytes)
      .def("num_allocated_blocks", &LayerType::numAllocatedBlocks);
}

/// Function for declaring a mesh type with a given name
template <typename MeshType>
void defineMeshClass(torch::Library& m, const std::string& name) {
  m.class_<MeshType>(name)
      .def(torch::init())
      .def("vertices", &MeshType::vertices)
      .def("triangles", &MeshType::triangles)
      .def("vertex_appearances", &MeshType::vertex_appearances);
}

TORCH_LIBRARY(pynvblox, m) {
  // NOTE: The order here is important. Classes have to be defined
  // (i.e. appear in this list) before methods that return them.

  m.class_<Constants>("Constants")
      .def(torch::init())
      .def("feature_array_num_elements", &Constants::featureArrayNumElements)
      .def("feature_array_element_size", &Constants::featureArrayElementSize)
      .def("esdf_unknown_distance", &Constants::kESDFUnknownDistance);

  defineVoxelBlockLayerClass<PyTsdfLayer>(m, "TsdfLayer");
  defineVoxelBlockLayerClass<PyColorLayer>(m, "ColorLayer");
  defineVoxelBlockLayerClass<PyFeatureLayer>(m, "FeatureLayer");

  defineMeshClass<PyColorMesh>(m, "ColorMesh");
  defineMeshClass<PyFeatureMesh>(m, "FeatureMesh");

  m.def("render_depth_image", &renderDepthImage)
      .def("render_depth_and_color_image", &renderDepthAndColorImage);

  m.class_<ProjectiveIntegratorParams>("ProjectiveIntegratorParams")
      .def(torch::init())
      .def("get_projective_integrator_max_integration_distance_m",
           &ProjectiveIntegratorParams::
               get_projective_integrator_max_integration_distance_m)
      .def("set_projective_integrator_max_integration_distance_m",
           &ProjectiveIntegratorParams::
               set_projective_integrator_max_integration_distance_m)
      .def("get_lidar_projective_integrator_max_integration_distance_m",
           &ProjectiveIntegratorParams::
               get_lidar_projective_integrator_max_integration_distance_m)
      .def("set_lidar_projective_integrator_max_integration_distance_m",
           &ProjectiveIntegratorParams::
               set_lidar_projective_integrator_max_integration_distance_m)
      .def("get_projective_integrator_truncation_distance_vox",
           &ProjectiveIntegratorParams::
               get_projective_integrator_truncation_distance_vox)
      .def("set_projective_integrator_truncation_distance_vox",
           &ProjectiveIntegratorParams::
               set_projective_integrator_truncation_distance_vox)
      .def(
          "get_projective_integrator_weighting_mode",
          &ProjectiveIntegratorParams::get_projective_integrator_weighting_mode)
      .def(
          "set_projective_integrator_weighting_mode",
          &ProjectiveIntegratorParams::set_projective_integrator_weighting_mode)
      .def("get_projective_integrator_max_weight",
           &ProjectiveIntegratorParams::get_projective_integrator_max_weight)
      .def("set_projective_integrator_max_weight",
           &ProjectiveIntegratorParams::set_projective_integrator_max_weight)
      .def("get_projective_tsdf_integrator_invalid_depth_decay_factor",
           &ProjectiveIntegratorParams::
               get_projective_tsdf_integrator_invalid_depth_decay_factor)
      .def("set_projective_tsdf_integrator_invalid_depth_decay_factor",
           &ProjectiveIntegratorParams::
               set_projective_tsdf_integrator_invalid_depth_decay_factor)
      .def("get_projective_appearance_integrator_measurement_weight",
           &ProjectiveIntegratorParams::
               get_projective_appearance_integrator_measurement_weight)
      .def("set_projective_appearance_integrator_measurement_weight",
           &ProjectiveIntegratorParams::
               set_projective_appearance_integrator_measurement_weight);

  m.class_<MeshIntegratorParams>("MeshIntegratorParams")
      .def(torch::init())
      .def("get_mesh_integrator_min_weight",
           &MeshIntegratorParams::get_mesh_integrator_min_weight)
      .def("set_mesh_integrator_min_weight",
           &MeshIntegratorParams::set_mesh_integrator_min_weight)
      .def("get_mesh_integrator_weld_vertices",
           &MeshIntegratorParams::get_mesh_integrator_weld_vertices)
      .def("set_mesh_integrator_weld_vertices",
           &MeshIntegratorParams::set_mesh_integrator_weld_vertices);

  m.class_<DecayIntegratorBaseParams>("DecayIntegratorBaseParams")
      .def(torch::init())
      .def("get_decay_integrator_deallocate_decayed_blocks",
           &DecayIntegratorBaseParams::
               get_decay_integrator_deallocate_decayed_blocks)
      .def("set_decay_integrator_deallocate_decayed_blocks",
           &DecayIntegratorBaseParams::
               set_decay_integrator_deallocate_decayed_blocks);

  m.class_<TsdfDecayIntegratorParams>("TsdfDecayIntegratorParams")
      .def(torch::init())
      .def("get_tsdf_decay_factor",
           &TsdfDecayIntegratorParams::get_tsdf_decay_factor)
      .def("set_tsdf_decay_factor",
           &TsdfDecayIntegratorParams::set_tsdf_decay_factor)
      .def("get_tsdf_decayed_weight_threshold",
           &TsdfDecayIntegratorParams::get_tsdf_decayed_weight_threshold)
      .def("set_tsdf_decayed_weight_threshold",
           &TsdfDecayIntegratorParams::set_tsdf_decayed_weight_threshold)
      .def("get_tsdf_set_free_distance_on_decayed",
           &TsdfDecayIntegratorParams::get_tsdf_set_free_distance_on_decayed)
      .def("set_tsdf_set_free_distance_on_decayed",
           &TsdfDecayIntegratorParams::set_tsdf_set_free_distance_on_decayed)
      .def("get_tsdf_decayed_free_distance_vox",
           &TsdfDecayIntegratorParams::get_tsdf_decayed_free_distance_vox)
      .def("set_tsdf_decayed_free_distance_vox",
           &TsdfDecayIntegratorParams::set_tsdf_decayed_free_distance_vox);

  m.class_<OccupancyDecayIntegratorParams>("OccupancyDecayIntegratorParams")
      .def(torch::init())
      .def("get_free_region_decay_probability",
           &OccupancyDecayIntegratorParams::get_free_region_decay_probability)
      .def("set_free_region_decay_probability",
           &OccupancyDecayIntegratorParams::set_free_region_decay_probability)
      .def("get_occupied_region_decay_probability",
           &OccupancyDecayIntegratorParams::
               get_occupied_region_decay_probability)
      .def("set_occupied_region_decay_probability",
           &OccupancyDecayIntegratorParams::
               set_occupied_region_decay_probability)
      .def("get_occupancy_decay_to_free",
           &OccupancyDecayIntegratorParams::get_occupancy_decay_to_free)
      .def("set_occupancy_decay_to_free",
           &OccupancyDecayIntegratorParams::set_occupancy_decay_to_free);

  m.class_<EsdfIntegratorParams>("EsdfIntegratorParams")
      .def(torch::init())
      .def("get_esdf_integrator_max_distance_m",
           &EsdfIntegratorParams::get_esdf_integrator_max_distance_m)
      .def("set_esdf_integrator_max_distance_m",
           &EsdfIntegratorParams::set_esdf_integrator_max_distance_m)
      .def("get_esdf_integrator_min_weight",
           &EsdfIntegratorParams::get_esdf_integrator_min_weight)
      .def("set_esdf_integrator_min_weight",
           &EsdfIntegratorParams::set_esdf_integrator_min_weight)
      .def("get_esdf_integrator_max_site_distance_vox",
           &EsdfIntegratorParams::get_esdf_integrator_max_site_distance_vox)
      .def("set_esdf_integrator_max_site_distance_vox",
           &EsdfIntegratorParams::set_esdf_integrator_max_site_distance_vox)
      .def("get_esdf_slice_min_height",
           &EsdfIntegratorParams::get_esdf_slice_min_height)
      .def("set_esdf_slice_min_height",
           &EsdfIntegratorParams::set_esdf_slice_min_height)
      .def("get_esdf_slice_max_height",
           &EsdfIntegratorParams::get_esdf_slice_max_height)
      .def("set_esdf_slice_max_height",
           &EsdfIntegratorParams::set_esdf_slice_max_height)
      .def("get_esdf_slice_height",
           &EsdfIntegratorParams::get_esdf_slice_height)
      .def("set_esdf_slice_height",
           &EsdfIntegratorParams::set_esdf_slice_height)
      .def("get_slice_height_above_plane_m",
           &EsdfIntegratorParams::get_slice_height_above_plane_m)
      .def("set_slice_height_above_plane_m",
           &EsdfIntegratorParams::set_slice_height_above_plane_m)
      .def("get_slice_height_thickness_m",
           &EsdfIntegratorParams::get_slice_height_thickness_m)
      .def("set_slice_height_thickness_m",
           &EsdfIntegratorParams::set_slice_height_thickness_m);

  m.class_<ViewCalculatorParams>("ViewCalculatorParams")
      .def(torch::init())
      .def("get_raycast_subsampling_factor",
           &ViewCalculatorParams::get_raycast_subsampling_factor)
      .def("set_raycast_subsampling_factor",
           &ViewCalculatorParams::set_raycast_subsampling_factor)
      .def("get_workspace_bounds_type",
           &ViewCalculatorParams::get_workspace_bounds_type)
      .def("set_workspace_bounds_type",
           &ViewCalculatorParams::set_workspace_bounds_type)
      .def("get_workspace_bounds_min_height_m",
           &ViewCalculatorParams::get_workspace_bounds_min_height_m)
      .def("set_workspace_bounds_min_height_m",
           &ViewCalculatorParams::set_workspace_bounds_min_height_m)
      .def("get_workspace_bounds_max_height_m",
           &ViewCalculatorParams::get_workspace_bounds_max_height_m)
      .def("set_workspace_bounds_max_height_m",
           &ViewCalculatorParams::set_workspace_bounds_max_height_m)
      .def("get_workspace_bounds_min_corner_x_m",
           &ViewCalculatorParams::get_workspace_bounds_min_corner_x_m)
      .def("set_workspace_bounds_min_corner_x_m",
           &ViewCalculatorParams::set_workspace_bounds_min_corner_x_m)
      .def("get_workspace_bounds_max_corner_x_m",
           &ViewCalculatorParams::get_workspace_bounds_max_corner_x_m)
      .def("set_workspace_bounds_max_corner_x_m",
           &ViewCalculatorParams::set_workspace_bounds_max_corner_x_m)
      .def("get_workspace_bounds_min_corner_y_m",
           &ViewCalculatorParams::get_workspace_bounds_min_corner_y_m)
      .def("set_workspace_bounds_min_corner_y_m",
           &ViewCalculatorParams::set_workspace_bounds_min_corner_y_m)
      .def("get_workspace_bounds_max_corner_y_m",
           &ViewCalculatorParams::get_workspace_bounds_max_corner_y_m)
      .def("set_workspace_bounds_max_corner_y_m",
           &ViewCalculatorParams::set_workspace_bounds_max_corner_y_m);

  m.class_<BlockMemoryPoolParams>("BlockMemoryPoolParams")
      .def(torch::init())
      .def("get_num_preallocated_blocks",
           &BlockMemoryPoolParams::get_num_preallocated_blocks)
      .def("set_num_preallocated_blocks",
           &BlockMemoryPoolParams::set_num_preallocated_blocks)
      .def("get_expansion_factor", &BlockMemoryPoolParams::get_expansion_factor)
      .def("set_expansion_factor",
           &BlockMemoryPoolParams::set_expansion_factor);

  m.class_<MapperParams>("MapperParams")
      .def(torch::init())
      .def("get_projective_integrator_params",
           &MapperParams::get_projective_integrator_params)
      .def("set_projective_integrator_params",
           &MapperParams::set_projective_integrator_params)
      .def("get_mesh_integrator_params",
           &MapperParams::get_mesh_integrator_params)
      .def("set_mesh_integrator_params",
           &MapperParams::set_mesh_integrator_params)
      .def("get_decay_integrator_base_params",
           &MapperParams::get_decay_integrator_base_params)
      .def("set_decay_integrator_base_params",
           &MapperParams::set_decay_integrator_base_params)
      .def("get_tsdf_decay_integrator_params",
           &MapperParams::get_tsdf_decay_integrator_params)
      .def("set_tsdf_decay_integrator_params",
           &MapperParams::set_tsdf_decay_integrator_params)
      .def("get_occupancy_decay_integrator_params",
           &MapperParams::get_occupancy_decay_integrator_params)
      .def("set_occupancy_decay_integrator_params",
           &MapperParams::set_occupancy_decay_integrator_params)
      .def("get_esdf_integrator_params",
           &MapperParams::get_esdf_integrator_params)
      .def("set_esdf_integrator_params",
           &MapperParams::set_esdf_integrator_params)
      .def("get_view_calculator_params",
           &MapperParams::get_view_calculator_params)
      .def("set_view_calculator_params",
           &MapperParams::set_view_calculator_params)
      .def("get_block_memory_pool_params",
           &MapperParams::get_block_memory_pool_params)
      .def("set_block_memory_pool_params",
           &MapperParams::set_block_memory_pool_params);

  m.class_<Mapper>("Mapper")
      .def(torch::init<std::vector<double>, std::vector<std::string>,
                       c10::intrusive_ptr<MapperParams>>())
      // Mapping methods
      .def("integrate_depth", &Mapper::integrateDepth)
      .def("integrate_color", &Mapper::integrateColor)
      .def("integrate_features", &Mapper::integrateFeatures)
      .def("update_esdf", &Mapper::updateEsdf)
      .def("update_color_mesh", &Mapper::updateColorMesh)
      .def("update_feature_mesh", &Mapper::updateFeatureMesh)
      .def("clear", &Mapper::clear)
      // Parmeters
      .def("params", &Mapper::getMapperParams)
      // Layer access
      .def("tsdf_layer", &Mapper::tsdf_layer)
      .def("color_layer", &Mapper::color_layer)
      .def("feature_layer", &Mapper::feature_layer)
      // Decay methods
      .def("decay_tsdf", &Mapper::decayTsdf)
      .def("decay_occupancy", &Mapper::decayOccupancy)
      // Rendering methods
      .def("render_depth_image", &Mapper::renderDepthImage)
      .def("render_depth_and_color_image", &Mapper::renderDepthAndColorImage)
      // Query methods
      .def("query_features", &Mapper::queryFeatures)
      // TODO(dtingdahl) add query_multi_features

      .def("query_esdf", &Mapper::queryEsdf)
      .def("query_multi_esdf", &Mapper::queryMultiEsdf)
      .def("query_tsdf", &Mapper::queryTsdf)
      .def("query_multi_tsdf", &Mapper::queryMultiTsdf)
      .def("query_multi_occupancy", &Mapper::queryMultiOccupancy)
      // Access methods
      .def("get_color_mesh", &Mapper::getColorMesh)
      .def("get_feature_mesh", &Mapper::getFeatureMesh)
      // File methods
      .def("output_color_mesh_ply", &Mapper::outputColorMeshPly)
      .def("load_from_file", &Mapper::loadFromFile)
      .def("output_blox_map", &Mapper::outputBloxMap)
      // Attributes
      .def("num_mappers", &Mapper::getNumMappers)
      // Benchmarking
      .def("print_timing", &Mapper::printTiming);

  m.class_<Scene>("Scene")
      .def(torch::init())
      .def("set_aabb", &Scene::setAABB)
      .def("get_aabb", &Scene::getAABB)
      .def("add_plane_boundaries", &Scene::addPlaneBoundaries)
      .def("add_ground_level", &Scene::addGroundLevel)
      .def("add_ceiling", &Scene::addCeiling)
      .def("add_primitive", &Scene::addPrimitive)
      .def("get_primitives_type_list", &Scene::getPrimitiveTypesList)
      .def("create_dummy_map", &Scene::createDummyMap)
      .def("to_mapper", &Scene::toMapper);
}

}  // namespace pynvblox
