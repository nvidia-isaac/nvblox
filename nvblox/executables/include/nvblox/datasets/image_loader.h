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

#include <deque>
#include <future>
#include <string>

#include "nvblox/core/image.h"
#include "nvblox/core/types.h"

namespace nvblox {
namespace datasets {

// Loads a depth image encoded as a 16-bit PNG, by default we assume that the
// depth is expressed in mm, and we converts it to a float with meters (through
// a multiplication with 1.0/1000.0).
constexpr float kDefaultUintDepthScaleFactor = 1.0f / 1000.0f;
constexpr float kDefaultUintDepthScaleOffset = 0.0f;

bool load16BitDepthImage(
    const std::string& filename, DepthImage* depth_frame_ptr,
    MemoryType memory_type = kDefaultImageMemoryType,
    const float scaling_factor = kDefaultUintDepthScaleFactor,
    const float scaling_offset = 0.0f);

bool load8BitColorImage(const std::string& filename,
                        ColorImage* color_image_ptr,
                        MemoryType memory_type = kDefaultImageMemoryType);

using IndexToFilepathFunction = std::function<std::string(int image_idx)>;

// Single-threaded image loader
template <typename ImageType>
class ImageLoader {
 public:
  ImageLoader(IndexToFilepathFunction index_to_filepath,
              MemoryType memory_type = kDefaultImageMemoryType,
              float depth_image_scaling_factor = kDefaultUintDepthScaleFactor,
              float depth_image_scaling_offset = kDefaultUintDepthScaleOffset)
      : index_to_filepath_(index_to_filepath),
        memory_type_(memory_type),
        depth_image_scaling_factor_(depth_image_scaling_factor),
        depth_image_scaling_offset_(depth_image_scaling_offset) {}
  virtual ~ImageLoader() {}

  virtual bool getNextImage(ImageType* image_ptr);

 protected:
  bool getImage(int image_idx, ImageType*);

  int image_idx_ = 0;
  const IndexToFilepathFunction index_to_filepath_;
  const MemoryType memory_type_;

  // Note(alexmillane): Only used for depth image loading. Ignored for color;
  const float depth_image_scaling_factor_;
  const float depth_image_scaling_offset_;
};

// Multi-threaded image loader
template <typename ImageType>
using ImageOptional = std::pair<bool, ImageType>;

// Multi-threaded image loader
template <typename ImageType>
class MultiThreadedImageLoader : public ImageLoader<ImageType> {
 public:
  MultiThreadedImageLoader(
      IndexToFilepathFunction index_to_filepath, int num_threads,
      MemoryType memory_type = kDefaultImageMemoryType,
      float depth_image_scaling_factor = kDefaultUintDepthScaleFactor,
      float depth_image_scaling_offset = kDefaultUintDepthScaleOffset);
  ~MultiThreadedImageLoader();

  bool getNextImage(ImageType* image_ptr) override;

 protected:
  void initLoadQueue();
  void emptyLoadQueue();
  void addNextImageToQueue();
  ImageOptional<ImageType> getImageAsOptional(int image_idx);

  const int num_threads_;
  std::deque<std::future<ImageOptional<ImageType>>> load_queue_;
};

// Factory Function
template <typename ImageType>
std::unique_ptr<ImageLoader<ImageType>> createImageLoader(
    IndexToFilepathFunction index_to_path_function,
    const bool multithreaded = true,
    const float depth_image_scaling_factor = kDefaultUintDepthScaleFactor,
    const float depth_image_scaling_offset = kDefaultUintDepthScaleOffset);

}  // namespace datasets
}  // namespace nvblox

#include "nvblox/datasets/impl/image_loader_impl.h"
