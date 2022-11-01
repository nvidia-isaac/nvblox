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

namespace nvblox {
namespace datasets {

// NOTE(jjiao): this function loads image depending on the specified ImageType
template <typename ImageType>
bool ImageLoader<ImageType>::getNextImage(ImageType* image_ptr) {
  bool res = getImage(image_idx_, image_ptr);
  ++image_idx_;
  return res;
}

template <typename ImageType>
MultiThreadedImageLoader<ImageType>::MultiThreadedImageLoader(
    IndexToFilepathFunction index_to_filepath, int num_threads,
    MemoryType memory_type, float depth_image_scaling_factor,
    float depth_image_scaling_offset)
    : ImageLoader<ImageType>(index_to_filepath, memory_type,
                             depth_image_scaling_factor,
                             depth_image_scaling_offset),
      num_threads_(num_threads) {
  initLoadQueue();
}

template <typename ImageType>
MultiThreadedImageLoader<ImageType>::~MultiThreadedImageLoader() {
  emptyLoadQueue();
}

template <typename ImageType>
bool MultiThreadedImageLoader<ImageType>::getNextImage(ImageType* image_ptr) {
  CHECK_NOTNULL(image_ptr);
  auto images_optional_future = std::move(load_queue_.front());
  ImageOptional<ImageType> image_optional =
      std::move(images_optional_future.get());
  if (image_optional.first) {
    *image_ptr = std::move(image_optional.second);
  }
  load_queue_.pop_front();
  addNextImageToQueue();
  return image_optional.first;
}

template <typename ImageType>
void MultiThreadedImageLoader<ImageType>::initLoadQueue() {
  for (int i = 0; i < num_threads_; i++) {
    addNextImageToQueue();
  }
}

template <typename ImageType>
void MultiThreadedImageLoader<ImageType>::emptyLoadQueue() {
  while (!load_queue_.empty()) {
    load_queue_.pop_front();
  }
}

template <typename ImageType>
void MultiThreadedImageLoader<ImageType>::addNextImageToQueue() {
  load_queue_.push_back(
      std::async(std::launch::async,
                 &MultiThreadedImageLoader<ImageType>::getImageAsOptional, this,
                 this->image_idx_));
  ++this->image_idx_;
}

template <typename ImageType>
ImageOptional<ImageType>
MultiThreadedImageLoader<ImageType>::getImageAsOptional(int image_idx) {
  ImageType image;
  bool success_flag = ImageLoader<ImageType>::getImage(image_idx, &image);
  return {success_flag, std::move(image)};
}

// Factory Function
template <typename ImageType>
std::unique_ptr<ImageLoader<ImageType>> createImageLoader(
    IndexToFilepathFunction index_to_path_function, const bool multithreaded,
    const float depth_image_scaling_factor,
    const float depth_image_scaling_offset) {
  if (multithreaded) {
    // NOTE(alexmillane): On my desktop the performance of threaded image
    // loading seems to saturate at around 6 threads. On machines with less
    // cores (I have 12 physical) presumably the saturation point will be less.
    constexpr unsigned int kMaxLoadingThreads = 6;
    unsigned int num_loading_threads =
        std::min(kMaxLoadingThreads, std::thread::hardware_concurrency() / 2);
    CHECK_GT(num_loading_threads, 0)
        << "std::thread::hardware_concurrency() returned 0";
    LOG(INFO) << "Using " << num_loading_threads
              << " threads for loading images.";
    return std::make_unique<MultiThreadedImageLoader<ImageType>>(
        index_to_path_function, kMaxLoadingThreads, MemoryType::kDevice,
        depth_image_scaling_factor, depth_image_scaling_offset);
  }
  return std::make_unique<ImageLoader<ImageType>>(
      index_to_path_function, MemoryType::kDevice, depth_image_scaling_factor,
      depth_image_scaling_offset);
}

}  // namespace datasets
}  // namespace nvblox