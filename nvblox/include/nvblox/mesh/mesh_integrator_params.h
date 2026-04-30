/*
Copyright 2024 NVIDIA CORPORATION

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
#pragma once

#include "nvblox/utils/params.h"

namespace nvblox {
constexpr Param<float>::Description kMeshIntegratorMinWeightParamDesc{
    "mesh_integrator_min_weight", 1e-4,
    "Minimum weight of the TSDF to consider for inclusion in the mesh."};
constexpr Param<bool>::Description kMeshIntegratorWeldVerticesParamDesc{
    "mesh_integrator_weld_vertices", true,
    "Whether to weld identical vertices together in the mesh."};
constexpr Param<int>::Description kMeshIntegratorMaxFlatMeshTrianglesParamDesc{
    "mesh_integrator_max_flat_mesh_triangles", 2000000,
    "Initial max triangles for flat mesh buffer. Auto-grows if exceeded."};

struct MeshIntegratorParams {
  Param<float> mesh_integrator_min_weight{kMeshIntegratorMinWeightParamDesc};
  Param<bool> mesh_integrator_weld_vertices{
      kMeshIntegratorWeldVerticesParamDesc};
  Param<int> mesh_integrator_max_flat_mesh_triangles{
      kMeshIntegratorMaxFlatMeshTrianglesParamDesc};
};

}  // namespace nvblox
