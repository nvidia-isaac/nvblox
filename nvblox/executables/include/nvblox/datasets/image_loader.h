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

#include "nvblox/core/types.h"
#include "nvblox/sensors/image.h"

namespace nvblox {
namespace datasets {

// Loads a depth image encoded as a 16-bit PNG, by default we assume that the
// depth is expressed in mm, and we converts it to a float with meters (through
// a multiplication with 1.0/1000.0).
constexpr float kDefaultUintDepthScaleFactor = 1.0f / 1000.0f;
bool load16BitDepthImage(
    const std::string& filename, DepthImage* depth_frame_ptr,
    const float scaling_factor = kDefaultUintDepthScaleFactor);
bool load8BitColorImage(const std::string& filename,
                        ColorImage* color_image_ptr);

using IndexToFilepathFunction = std::function<std::string(int image_idx)>;

// Single-threaded image loader
template <typename ImageType>
class ImageLoader {
 public:
  ImageLoader(IndexToFilepathFunction index_to_filepath,
              float depth_image_scaling_factor = kDefaultUintDepthScaleFactor)
      : index_to_filepath_(index_to_filepath),
        depth_image_scaling_factor_(depth_image_scaling_factor) {}
  virtual ~ImageLoader() {}

  virtual bool getNextImage(ImageType* image_ptr);

 protected:
  bool getImage(int image_idx, ImageType*);

  int image_idx_ = 0;
  const IndexToFilepathFunction index_to_filepath_;

  // Note(alexmillane): Only used for depth image loading. Ignored for color;
  const float depth_image_scaling_factor_;
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
      float depth_image_scaling_factor = kDefaultUintDepthScaleFactor);
  ~MultiThreadedImageLoader();

  bool getNextImage(ImageType* image_ptr) override;

 protected:
  void initLoadQueue();
  void emptyLoadQueue();
  void addNextImageToQueue(MemoryType memory_type);
  ImageOptional<ImageType> getImageAsOptional(int image_idx,
                                              MemoryType memory_type);

  const int num_threads_;
  std::deque<std::future<ImageOptional<ImageType>>> load_queue_;
};

// Factory Function
template <typename ImageType>
std::unique_ptr<ImageLoader<ImageType>> createImageLoader(
    IndexToFilepathFunction index_to_path_function,
    const bool multithreaded = true,
    const float depth_image_scaling_factor = kDefaultUintDepthScaleFactor);

}  // namespace datasets
}  // namespace nvblox

#include "nvblox/datasets/impl/image_loader_impl.h"
