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

#include <cuda_runtime.h>

#include "nvblox/core/color.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_ptr.h"

namespace nvblox {

// Row-major access to images (on host and device)
namespace image {

template <typename ElementType>
__host__ __device__ inline ElementType access(int row_idx, int col_idx,
                                              int cols,
                                              const ElementType* data) {
  return data[row_idx * cols + col_idx];
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int row_idx, int col_idx,
                                               int cols, ElementType* data) {
  return data[row_idx * cols + col_idx];
}
template <typename ElementType>
__host__ __device__ inline ElementType access(int linear_idx,
                                              const ElementType* data) {
  return data[linear_idx];
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int linear_idx,
                                               ElementType* data) {
  return data[linear_idx];
}

}  // namespace image

// Row-major image.
// - Note that in a row-major image, rows follow one another in linear memory,
// which means the column index varied between subsequent elements.
// - Images use corner based indexing such that the pixel with index (0,0) is
// centered at (0.5px,0.5px) and spans (0.0px,0.0px) to (1.0px,1.0px).
// - Points on the image plane are defined as: u_px = (u_px.x(), u_px.y()) =
// (col_idx, row_idx) in pixels.

constexpr MemoryType kDefaultImageMemoryType = MemoryType::kDevice;

template <typename _ElementType>
class Image {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;

  Image(int rows, int cols, MemoryType memory_type = kDefaultImageMemoryType)
      : rows_(rows),
        cols_(cols),
        memory_type_(memory_type),
        data_(make_unified<ElementType[]>(static_cast<size_t>(rows * cols),
                                          memory_type_)) {}
  Image(MemoryType memory_type = kDefaultImageMemoryType)
      : rows_(0), cols_(0), memory_type_(memory_type) {}
  Image(Image&& other);
  Image& operator=(Image&& other);

  // Deep copy constructor (second can be used to transition memory type)
  explicit Image(const Image& other);
  Image(const Image& other, MemoryType memory_type);
  Image& operator=(const Image& other);

  // Prefetch the data to the gpu (only makes sense for kUnified images)
  void toGPU() const;

  // Attributes
  inline int cols() const { return cols_; }
  inline int rows() const { return rows_; }
  inline int numel() const { return cols_ * rows_; }
  inline int width() const { return cols_; }
  inline int height() const { return rows_; }
  inline MemoryType memory_type() const { return memory_type_; }

  // Access
  // NOTE(alexmillane): The guard-rails are off here. If you declare a kDevice
  // Image and try to access its data, you will get undefined behaviour.
  inline ElementType operator()(const int row_idx, const int col_idx) const {
    return image::access(row_idx, col_idx, cols_, data_.get());
  }
  inline ElementType& operator()(const int row_idx, const int col_idx) {
    return image::access(row_idx, col_idx, cols_, data_.get());
  }
  inline ElementType operator()(const int linear_idx) const {
    return image::access(linear_idx, data_.get());
  }
  inline ElementType& operator()(const int linear_idx) {
    return image::access(linear_idx, data_.get());
  }

  // Raw pointer access
  inline ElementType* dataPtr() { return data_.get(); }
  inline const ElementType* dataConstPtr() const { return data_.get(); }

  // Reset the contents of the image. Reallocate if the image got larger.
  void populateFromBuffer(int rows, int cols, const ElementType* buffer,
                          MemoryType memory_type = kDefaultImageMemoryType);

  // Set the image to 0.
  void setZero();

  // Factories
  static inline Image fromBuffer(
      int rows, int cols, const ElementType* buffer,
      MemoryType memory_type = kDefaultImageMemoryType);

 protected:
  int rows_;
  int cols_;
  MemoryType memory_type_;
  unified_ptr<ElementType[]> data_;
};

using DepthImage = Image<float>;
using ColorImage = Image<Color>;

// Image Reductions
namespace image {

float max(const DepthImage& image);
float min(const DepthImage& image);
std::pair<float, float> minmax(const DepthImage& image);
float maxGPU(const DepthImage& image);
float minGPU(const DepthImage& image);
std::pair<float, float> minmaxGPU(const DepthImage& image);
float maxCPU(const DepthImage& image);
float minCPU(const DepthImage& image);
std::pair<float, float> minmaxCPU(const DepthImage& image);

void elementWiseMinInPlace(const float constant, DepthImage* image);
void elementWiseMaxInPlace(const float constant, DepthImage* image);

template <typename ImageType>
void getDifferenceImageGPU(const ImageType& image_1, const ImageType& image_2,
                           ImageType* diff_image_ptr);

}  // namespace image

}  // namespace nvblox

#include "nvblox/core/impl/image_impl.h"
