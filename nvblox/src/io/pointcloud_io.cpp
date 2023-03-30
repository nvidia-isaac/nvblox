#include "nvblox/io/pointcloud_io.h"

namespace nvblox {
namespace io {

/// Specializations for the TSDF type.
template <>
bool outputVoxelLayerToPly(const TsdfLayer& layer,
                           const std::string& filename) {
  constexpr float kMinWeight = 0.1f;
  auto lambda = [&kMinWeight](const TsdfVoxel* voxel, float* distance) -> bool {
    *distance = voxel->distance;
    return voxel->weight > kMinWeight;
  };
  return outputVoxelLayerToPly<TsdfVoxel>(layer, filename, lambda);
}

/// Specialization for the ESDF type.
template <>
bool outputVoxelLayerToPly(const EsdfLayer& layer,
                           const std::string& filename) {
  const float voxel_size = layer.voxel_size();
  auto lambda = [&voxel_size](const EsdfVoxel* voxel, float* distance) -> bool {
    *distance = voxel_size * std::sqrt(voxel->squared_distance_vox);
    if (voxel->is_inside) {
      *distance = -*distance;
    }
    return voxel->observed;
  };
  return outputVoxelLayerToPly<EsdfVoxel>(layer, filename, lambda);
}

// Specialization for the TSDF type in cpp file.
// template <>
// bool outputVoxelLayerToPly(const TsdfLayer& layer, const std::string& filename);

// Specialization for the ESDF type in cpp file.
// template <>
// bool outputVoxelLayerToPly(const EsdfLayer& layer, const std::string& filename);

} // namespace io
} // namespace nvblox
