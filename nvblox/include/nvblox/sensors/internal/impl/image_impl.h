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

namespace nvblox {

// Image (memory owning)
template <typename ElementType>
Image<ElementType>::Image(int rows, int cols, MemoryType memory_type)
    : ImageBase<ElementType>(rows, cols),
      memory_type_(memory_type),
      owned_data_(make_unified<ElementType[]>(static_cast<size_t>(rows * cols),
                                              memory_type_)) {
  CHECK_GT(rows, 0)
      << "Invalid size. Use another constructor to construct empty image.";
  CHECK_GT(cols, 0)
      << "Invalid size. Use another constructor to construct empty image.";
  ImageBase<ElementType>::data_ = owned_data_.get();
}

template <typename ElementType>
Image<ElementType>::Image(Image<ElementType>&& other)
    : ImageBase<ElementType>(other.rows_, other.cols_, other.data_),
      memory_type_(other.memory_type()),
      owned_data_(std::move(other.owned_data_)) {}

template <typename ElementType>
Image<ElementType>& Image<ElementType>::operator=(Image<ElementType>&& other) {
  this->data_ = other.data_;
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  memory_type_ = other.memory_type_;
  owned_data_ = std::move(other.owned_data_);
  return *this;
}

template <typename ElementType>
void Image<ElementType>::copyFrom(const Image& other) {
  copyFromAsync(other, CudaStreamOwning());
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(const Image& other,
                                       const CudaStream cuda_stream) {
  copyFromAsync(other.rows(), other.cols(), other.dataConstPtr(), cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::copyFrom(const size_t rows, const size_t cols,
                                  const ElementType* const buffer) {
  copyFromAsync(rows, cols, buffer, CudaStreamOwning());
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(const size_t rows, const size_t cols,
                                       const ElementType* const buffer,
                                       const CudaStream cuda_stream) {
  if (!owned_data_ || this->numel() < static_cast<int>(rows * cols)) {
    // We need to reallocate.
    owned_data_ = make_unified<ElementType[]>(static_cast<size_t>(rows * cols),
                                              memory_type_);
    this->data_ = owned_data_.get();
  }

  this->rows_ = rows;
  this->cols_ = cols;

  owned_data_.copyFromAsync(buffer, rows * cols, cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::setZero() {
  owned_data_.setZero();
}

// ImageView (shallow image)

// Wrap an existing buffer
template <typename ElementType>
ImageView<ElementType>::ImageView(int rows, int cols, ElementType* data)
    : ImageBase<ElementType>(rows, cols, data) {}

/// (Shallow) Copy
template <typename ElementType>
ImageView<ElementType>::ImageView(const ImageView& other)
    : ImageBase<ElementType>(other.rows_, other.cols_, other.data_) {}

template <typename ElementType>
ImageView<ElementType>& ImageView<ElementType>::operator=(
    const ImageView<ElementType>& other) {
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  this->data_ = other.data_;
  return *this;
}

/// Move (ImageView is a shallow copy so a move-construction is the same as
/// copy-construction)
template <typename ElementType>
ImageView<ElementType>::ImageView(ImageView&& other) : ImageView(other) {}

template <typename ElementType>
ImageView<ElementType>& ImageView<ElementType>::operator=(
    ImageView<ElementType>&& other) {
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  this->data_ = other.data_;
  return *this;
}

template <typename ElementType>
ImageView<ElementType>::ImageView(Image<ElementType>& image)
    : ImageView(image.rows(), image.cols(), image.dataPtr()) {}

template <typename ElementType>
ImageView<ElementType>::ImageView(const Image<ElementType_nonconst>& image)
    : ImageView(image.rows(), image.cols(), image.dataConstPtr()) {}

}  // namespace nvblox
