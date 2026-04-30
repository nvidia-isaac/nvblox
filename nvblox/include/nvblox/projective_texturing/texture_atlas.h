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
#include "nvblox/core/types.h"
#include "nvblox/projective_texturing/camera_view.h"
#include "nvblox/sensors/image.h"

namespace nvblox {

/// Texture atlas that packs one or more camera images into a single texture.
///
/// For single camera: the atlas is the camera image itself.
/// For multi-camera: images are packed side-by-side into a single texture.
/// Layout details are internal -- the ProjectiveTextureMapper uses the atlas
/// to compute atlas-space UVs directly.
class TextureAtlas {
 public:
  TextureAtlas() = default;

  /// Build the atlas by stitching all camera images into a single texture.
  /// @param views Camera views with images.
  /// @param stream CUDA stream for GPU operations.
  void buildAtlasAsync(const std::vector<CameraView>& views,
                       const CudaStream& stream);

  /// Get the stitched atlas image (single texture for renderer).
  const ColorImage& atlasImage() const { return atlas_; }

  /// Get the number of cameras in the atlas.
  int numCameras() const { return num_cameras_; }

  /// Get the atlas-space UV offset for a camera region.
  /// @param camera_index Camera index.
  /// @return UV offset (x_offset, y_offset) in [0, 1].
  Vector2f uvOffset(int camera_index) const;

  /// Get the atlas-space UV scale for a camera region.
  /// @param camera_index Camera index.
  /// @return UV scale (x_scale, y_scale) mapping [0,1] image UVs to atlas UVs.
  Vector2f uvScale(int camera_index) const;

 private:
  ColorImage atlas_{MemoryType::kDevice};
  int num_cameras_ = 0;

  // Per-camera layout info (used by ProjectiveTextureMapper)
  struct Region {
    Vector2f offset;  // UV offset in atlas
    Vector2f scale;   // UV scale in atlas
  };
  std::vector<Region> regions_;
};

}  // namespace nvblox
