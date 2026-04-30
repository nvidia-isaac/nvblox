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
#pragma once

#include <string>

#include "nvblox/geometry/plane.h"
#include "nvblox/map/common_names.h"
#include "nvblox/map/layer.h"
#include "nvblox/mesh/mesh_block.h"

namespace nvblox {
namespace io {

bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const std::string& filename);

bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const char* filename);

/// @brief Output a color mesh layer to PLY file with ground plane alignment.
///        The mesh will be transformed so that the ground plane is aligned to
///        z=0.
/// @param layer The color mesh layer to output.
/// @param filename The output PLY filename.
/// @param ground_plane Ground plane. The mesh will be transformed to align
///                     the plane to z=0.
/// @return True if successful, false otherwise.
bool outputColorMeshLayerToPly(const ColorMeshLayer& layer,
                               const std::string& filename,
                               const Plane& ground_plane);

}  // namespace io
}  // namespace nvblox
