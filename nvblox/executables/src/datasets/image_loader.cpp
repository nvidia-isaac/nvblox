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
#include "nvblox/datasets/image_loader.h"

#include <Eigen/Core>

#include "nvblox/utils/timing.h"
#include "nvblox/io/internal/thirdparty/stb_image.h"

namespace nvblox {
namespace datasets {

bool load16BitDepthImage(const std::string& filename,
                         DepthImage* depth_frame_ptr,
                         const float scale_factor) {
  CHECK_NOTNULL(depth_frame_ptr);
  timing::Timer stbi_timer("file_loading/depth_image/stbi");
  int width, height, num_channels;
  uint16_t* image_data =
      stbi_load_16(filename.c_str(), &width, &height, &num_channels, 0);
  stbi_timer.Stop();

  if (image_data == nullptr) {
    return false;
  }
  CHECK_EQ(num_channels, 1);

  // TODO(alexmillane): It's likely better to do this on the GPU.
  //                    Follow up: I measured and this scaling + cast takes
  //                    ~1ms. So only do this when 1ms is relevant.
  std::vector<float> float_image_data(height * width);
  for (size_t lin_idx = 0; lin_idx < float_image_data.size(); lin_idx++) {
    float_image_data[lin_idx] =
        static_cast<float>(image_data[lin_idx]) * scale_factor;
  }

  depth_frame_ptr->copyFrom(height, width, float_image_data.data());

  stbi_image_free(image_data);
  return true;
}

bool load8BitColorImage(const std::string& filename,
                        ColorImage* color_image_ptr) {
  CHECK_NOTNULL(color_image_ptr);
  timing::Timer stbi_timer("file_loading/color_image/stbi");
  int width, height, num_channels;
  uint8_t* image_data =
      stbi_load(filename.c_str(), &width, &height, &num_channels, 4);
  stbi_timer.Stop();

  if (image_data == nullptr) {
    return false;
  }
  // Currently we only support loading 3 channel (rgb) or 4 channel (rgba)
  // images.
  CHECK(num_channels == 3 || num_channels == 4);

  CHECK_EQ(sizeof(Color), 4 * sizeof(uint8_t))
      << "Color struct was padded by the compiler so image loading wont work.";

  color_image_ptr->copyFrom(height, width,
                            reinterpret_cast<Color*>(image_data));

  stbi_image_free(image_data);
  return true;
}

template <>
bool ImageLoader<DepthImage>::getImage(int image_idx, DepthImage* image_ptr) {
  CHECK_NOTNULL(image_ptr);
  bool res = load16BitDepthImage(index_to_filepath_(image_idx), image_ptr,
                                 depth_image_scaling_factor_);
  return res;
}

template <>
bool ImageLoader<ColorImage>::getImage(int image_idx, ColorImage* image_ptr) {
  CHECK_NOTNULL(image_ptr);
  bool res = load8BitColorImage(index_to_filepath_(image_idx), image_ptr);
  return res;
}

}  // namespace datasets
}  // namespace nvblox
