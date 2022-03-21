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

// Loads a depth image encoded as a 16-bit PNG, assumed to express the depth in
// mm, and converts it to a float with meters.
bool load16BitDepthImage(const std::string& filename,
                         DepthImage* depth_frame_ptr,
                         MemoryType memory_type = kDefaultImageMemoryType);
bool load8BitColorImage(const std::string& filename,
                        ColorImage* color_image_ptr,
                        MemoryType memory_type = kDefaultImageMemoryType);

using IndexToFilepathFunction = std::function<std::string(int image_idx)>;

// Single-threaded image loader
template <typename ImageType>
class ImageLoader {
 public:
  ImageLoader(IndexToFilepathFunction index_to_filepath,
              MemoryType memory_type = kDefaultImageMemoryType)
      : index_to_filepath_(index_to_filepath), memory_type_(memory_type) {}
  virtual ~ImageLoader() {}

  virtual bool getNextImage(ImageType* image_ptr);

 protected:
  bool getImage(int image_idx, ImageType*);

  int image_idx_ = 0;
  IndexToFilepathFunction index_to_filepath_;
  MemoryType memory_type_;
};

// Multi-threaded image loader
template <typename ImageType>
using ImageOptional = std::pair<bool, ImageType>;

// Multi-threaded image loader
template <typename ImageType>
class MultiThreadedImageLoader : public ImageLoader<ImageType> {
 public:
  MultiThreadedImageLoader(IndexToFilepathFunction index_to_filepath,
                           int num_threads,
                           MemoryType memory_type = kDefaultImageMemoryType);
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

}  // namespace datasets
}  // namespace nvblox

#include "nvblox/datasets/impl/image_loader_impl.h"
