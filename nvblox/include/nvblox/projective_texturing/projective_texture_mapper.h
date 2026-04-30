/*
Copyright 2026 NVIDIA CORPORATION

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

#include <vector>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/unified_vector.h"
#include "nvblox/mesh/mesh.h"
#include "nvblox/projective_texturing/camera_view.h"
#include "nvblox/projective_texturing/projective_texture_mapper_params.h"
#include "nvblox/projective_texturing/texture_atlas.h"

namespace nvblox {

/// Projects mesh vertices into camera images to compute atlas-space UVs.
///
/// This class performs projective texture mapping: for each mesh vertex,
/// it projects the 3D position into the camera image using the camera
/// matrix (intrinsics + extrinsics), producing UV coordinates that map
/// into the texture atlas.
///
/// Usage:
/// @code
/// ProjectiveTextureMapper texture_mapper;
/// texture_mapper.buildAtlasAsync({view0, view1}, stream);
/// texture_mapper.mapMesh(&mesh, stream);
/// // mesh.vertex_uvs is now populated
/// // texture_mapper.atlasImage() is the atlas image for the renderer
/// @endcode
///
/// For single-camera, the atlas is the camera image and UVs map directly
/// into it. For multi-camera, the atlas packs all images and UVs point
/// to the correct region.
class ProjectiveTextureMapper {
 public:
  ProjectiveTextureMapper() = default;

  /// Build the texture atlas from the given camera views.
  /// @param views Camera views with intrinsics, extrinsics, and images.
  /// @param stream CUDA stream.
  void buildAtlasAsync(std::vector<CameraView> views, const CudaStream& stream);

  /// Map mesh vertices to atlas-space UVs.
  /// Populates mesh->vertex_uvs with projected coordinates.
  /// Vertices outside all camera frustums get UV = (-1, -1).
  /// @note Requires aligned RGBD images in CameraView (depth aligned to
  /// color), and undistorted images.
  /// @param mesh ColorMesh with vertices populated (from marching cubes).
  /// @param stream CUDA stream.
  void mapMesh(ColorMesh* mesh, const CudaStream& stream);

  /// Get the texture atlas object.
  const TextureAtlas& textureAtlas() const { return atlas_; }

  /// Get the atlas image directly (convenience for passing to renderer).
  const ColorImage& atlasImage() const { return atlas_.atlasImage(); }

  /// Set the parameters of the projective texture mapper.
  /// @param params The struct containing the params.
  void setProjectiveTextureMapperParams(
      const ProjectiveTextureMapperParams& params) {
    params_ = params;
  }

  /// Get the parameters of the projective texture mapper.
  const ProjectiveTextureMapperParams& projective_texture_mapper_params()
      const {
    return params_;
  }

 private:
  TextureAtlas atlas_;
  std::vector<CameraView> views_;
  ProjectiveTextureMapperParams params_;

  /// Device buffer for CameraView data passed to the projection kernel.
  /// Resized as needed by mapMesh via copyFromAsync.
  unified_vector<CameraView> d_views_{MemoryType::kDevice};
};

}  // namespace nvblox
