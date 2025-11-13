/*
Copyright 2025 NVIDIA CORPORATION

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

#include "nvblox/integrators/occupancy_conversions.h"

#include <cmath>
#include <fstream>

#include <glog/logging.h>

#include "nvblox/core/types.h"
#include "nvblox/io/image_io.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace conversions {

void saveOccupancyGridAsPng(const std::string& png_path,
                            const float free_thresh,
                            const float occupied_thresh, const size_t height,
                            const size_t width,
                            const std::vector<int8_t>& occupancy_grid) {
  // Create image and then save it
  MonoImage image(height, width, MemoryType::kHost);

  int free_thresh_int = std::rint(free_thresh * 100.0);
  int occupied_thresh_int = std::rint(occupied_thresh * 100.0);

  // Fill the image
  for (size_t y = 0; y < height; y++) {
    for (size_t x = 0; x < width; x++) {
      int8_t map_cell = occupancy_grid[width * (height - y - 1) + x];

      uint8_t pixel;

      // Convert from OccupancyGrid format to png colors.
      if (map_cell < 0 || 100 < map_cell) {
        // Unknown
        pixel = uint8_t(kOccupancyGridPngUnknownValue);
      } else if (map_cell <= free_thresh_int) {
        // Free
        pixel = uint8_t(kOccupancyGridPngFreeValue);
      } else if (occupied_thresh_int <= map_cell) {
        // Occupied
        pixel = uint8_t(kOccupancyGridPngOccupiedValue);
      } else {
        // Everything else is also Unknown
        pixel = uint8_t(kOccupancyGridPngUnknownValue);
      }

      image(y, x) = pixel;
    }
  }

  // Write out the image
  io::writeToPng(png_path, image);
}

void saveOccupancyGridYaml(const std::string& yaml_path,
                           const std::string& image_name,
                           const float voxel_size, const float origin_x,
                           const float origin_y, const float free_thresh,
                           const float occupied_thresh) {
  // Write out the yaml
  std::ofstream yaml(yaml_path);
  CHECK(yaml.is_open()) << "Failed to open file: " << yaml_path;

  yaml << "image: \"" << image_name << "\"\n";
  yaml << "mode: \"trinary\"\n";
  yaml << "resolution: " << voxel_size << "\n";
  yaml << "origin: [" << origin_x << ", " << origin_y << ", 0]\n";
  yaml << "negate: 0\n";
  yaml << "occupied_thresh: " << occupied_thresh << "\n";
  yaml << "free_thresh: " << free_thresh << std::endl;
}

}  // namespace conversions
}  // namespace nvblox
