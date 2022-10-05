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

#include <algorithm>

#include "nvblox/core/cuda/image_cuda.h"
#include "nvblox/core/interpolation_2d.h"
#include "nvblox/io/csv.h"

namespace nvblox {

template <typename ElementType>
Image<ElementType>::Image(const Image<ElementType>& other)
    : Image(other, other.memory_type_) {
  LOG(WARNING) << "Deep copy of Image.";
};

template <typename ElementType>
Image<ElementType>::Image(const Image<ElementType>& other,
                          MemoryType memory_type)
    : rows_(other.rows()),
      cols_(other.cols()),
      memory_type_(memory_type),
      data_(make_unified<ElementType[]>(static_cast<size_t>(rows_ * cols_),
                                        memory_type)) {
  cuda::copy(rows_, cols_, other.data_.get(), data_.get());
}

template <typename ElementType>
Image<ElementType>::Image(Image<ElementType>&& other)
    : rows_(other.rows()),
      cols_(other.cols()),
      memory_type_(other.memory_type()),
      data_(std::move(other.data_)) {}

template <typename ElementType>
Image<ElementType>& Image<ElementType>::operator=(Image<ElementType>&& other) {
  rows_ = other.rows_;
  cols_ = other.cols_;
  memory_type_ = other.memory_type_;
  data_ = std::move(other.data_);
  return *this;
}

template <typename ElementType>
Image<ElementType>& Image<ElementType>::operator=(
    const Image<ElementType>& other) {
  LOG(WARNING) << "Deep copy of Image.";
  rows_ = other.rows_;
  cols_ = other.cols_;
  memory_type_ = other.memory_type_;
  data_ = make_unified<ElementType[]>(static_cast<size_t>(rows_ * cols_),
                                      memory_type_);
  cuda::copy(rows_, cols_, other.data_.get(), data_.get());
  return *this;
}

template <typename ElementType>
void Image<ElementType>::toGPU() const {
  CHECK(memory_type_ == MemoryType::kUnified)
      << "Called kUnified function on kDevice image.";
  cuda::toGPU(data_.get(), rows_, cols_);
}

template <typename ImageType>
ImageType fromBufferTemplate(int rows, int cols,
                             const typename ImageType::ElementType* buffer,
                             MemoryType memory_type) {
  ImageType image(rows, cols, memory_type);
  cuda::copy<typename ImageType::ElementType>(rows, cols, buffer,
                                              image.dataPtr());
  return image;
}

template <typename ElementType>
Image<ElementType> Image<ElementType>::fromBuffer(int rows, int cols,
                                                  const ElementType* buffer,
                                                  MemoryType memory_type) {
  return fromBufferTemplate<Image<ElementType>>(rows, cols, buffer,
                                                memory_type);
}

template <typename ElementType>
void Image<ElementType>::populateFromBuffer(int rows, int cols,
                                            const ElementType* buffer,
                                            MemoryType memory_type) {
  if (!data_ || numel() < rows * cols || memory_type != memory_type_) {
    // We need to reallocate.
    data_ = make_unified<ElementType[]>(static_cast<size_t>(rows * cols),
                                        memory_type);
  }
  rows_ = rows;
  cols_ = cols;
  memory_type_ = memory_type;
  cudaMemcpy(data_.get(), buffer, rows * cols * sizeof(ElementType),
             cudaMemcpyDefault);
}

template <typename ElementType>
void Image<ElementType>::setZero() {
  data_.setZero();
}

namespace image {

template <typename ImageType>
void getDifferenceImageGPU(const ImageType& image_1, const ImageType& image_2,
                           ImageType* diff_image_ptr) {
  CHECK_NOTNULL(diff_image_ptr);
  CHECK_EQ(image_1.rows(), image_2.rows());
  CHECK_EQ(image_1.cols(), image_2.cols());
  CHECK(image_1.memory_type() == MemoryType::kDevice ||
        image_1.memory_type() == MemoryType::kUnified);
  CHECK(image_2.memory_type() == MemoryType::kDevice ||
        image_2.memory_type() == MemoryType::kUnified);
  if (diff_image_ptr->rows() != image_1.rows() ||
      diff_image_ptr->cols() != image_1.cols()) {
    *diff_image_ptr =
        ImageType(image_1.rows(), image_1.cols(), image_1.memory_type());
  }
  cuda::diff(image_1.rows(), image_2.cols(), image_1.dataConstPtr(),
             image_2.dataConstPtr(), diff_image_ptr->dataPtr());
}

}  // namespace image
}  // namespace nvblox
