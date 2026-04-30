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
      owned_data_(rows * cols, memory_type) {
  CHECK_GT(rows, 0)
      << "Invalid size. Use another constructor to construct empty image.";
  CHECK_GT(cols, 0)
      << "Invalid size. Use another constructor to construct empty image.";
  ImageBase<ElementType>::data_ = owned_data_.data();
}

template <typename ElementType>
Image<ElementType>::Image(Image<ElementType>&& other)
    : ImageBase<ElementType>(other.rows_, other.cols_, other.stride_bytes(),
                             other.num_elements_per_pixel_, other.data_),
      memory_type_(other.memory_type()),
      owned_data_(std::move(other.owned_data_)) {
  other.data_ = nullptr;
  other.rows_ = 0;
  other.cols_ = 0;
  other.stride_num_elements_ = 0;
  other.num_elements_per_pixel_ = 1;
}

template <typename ElementType>
Image<ElementType>& Image<ElementType>::operator=(Image<ElementType>&& other) {
  this->data_ = other.data_;
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  this->stride_num_elements_ = other.stride_num_elements_;
  this->num_elements_per_pixel_ = other.num_elements_per_pixel_;
  memory_type_ = other.memory_type_;
  owned_data_ = std::move(other.owned_data_);

  other.data_ = nullptr;
  other.rows_ = 0;
  other.cols_ = 0;
  other.stride_num_elements_ = 0;
  other.num_elements_per_pixel_ = 1;
  return *this;
}

template <typename ElementType>
void Image<ElementType>::copyFrom(const ImageBase<ElementType>& other) {
  copyFromAsync(other, CudaStreamOwning());
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(
    const ImageBase<const ElementType>& other, const CudaStream& cuda_stream) {
  copyFromAsync(other.rows(), other.cols(), other.stride_num_elements(),
                other.num_elements_per_pixel(), other.dataConstPtr(),
                cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(const ImageBase<ElementType>& other,
                                       const CudaStream& cuda_stream) {
  copyFromAsync(other.rows(), other.cols(), other.stride_num_elements(),
                other.num_elements_per_pixel(), other.dataConstPtr(),
                cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::copyFrom(const size_t rows, const size_t cols,
                                  const ElementType* const buffer) {
  copyFrom(rows, cols, cols, 1, buffer);
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(const size_t rows, const size_t cols,
                                       const ElementType* const buffer,
                                       const CudaStream& cuda_stream) {
  copyFromAsync(rows, cols, cols, 1, buffer, cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::copyFrom(const size_t rows, const size_t cols,
                                  const size_t stride_num_elements,
                                  const size_t num_elements_per_pixel,
                                  const ElementType* const buffer) {
  copyFromAsync(rows, cols, stride_num_elements, num_elements_per_pixel, buffer,
                CudaStreamOwning());
}

template <typename ElementType>
void Image<ElementType>::copyFromAsync(const size_t rows, const size_t cols,
                                       const size_t stride_num_elements,
                                       const size_t num_elements_per_pixel,
                                       const ElementType* const buffer,
                                       const CudaStream& cuda_stream) {
  this->rows_ = rows;
  this->cols_ = cols;
  this->stride_num_elements_ = stride_num_elements;
  this->num_elements_per_pixel_ = num_elements_per_pixel;

  // TODO(dtingdahl) use strided memcpy for efficiency
  owned_data_.copyFromAsync(
      buffer, rows * stride_num_elements * num_elements_per_pixel, cuda_stream);
  ImageBase<ElementType>::data_ = owned_data_.data();
}

template <typename ElementType>
void Image<ElementType>::copyToAsync(ElementType* buffer,
                                     const CudaStream& cuda_stream) const {
  CHECK_NOTNULL(buffer);
  owned_data_.copyToAsync(static_cast<ElementType*>(buffer), cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::copyTo(ElementType* buffer) const {
  copyToAsync(buffer, CudaStreamOwning());
}

template <typename ElementType>
void Image<ElementType>::setZeroAsync(const CudaStream& cuda_stream) {
  owned_data_.setZeroAsync(cuda_stream);
}

template <typename ElementType>
void Image<ElementType>::resizeAsync(const size_t rows, const size_t cols,
                                     const CudaStream& cuda_stream) {
  this->rows_ = rows;
  this->cols_ = cols;
  this->stride_num_elements_ = cols;

  owned_data_.resizeAsync(rows * cols, cuda_stream);
  ImageBase<ElementType>::data_ = owned_data_.data();
}

// ImageView (shallow image)

// Wrap an existing buffer
template <typename ElementType>
template <typename OtherElementType>
ImageView<ElementType>::ImageView(int rows, int cols, OtherElementType* data)
    : ImageBase<ElementType>(rows, cols, data) {}

// Wrap an existing buffer with stride
template <typename ElementType>
template <typename OtherElementType>
ImageView<ElementType>::ImageView(int rows, int cols, int stride_bytes,
                                  int num_elements_per_pixel,
                                  OtherElementType* data)
    : ImageBase<ElementType>(rows, cols, stride_bytes, num_elements_per_pixel,
                             data) {}

/// (Shallow) Copy
template <typename ElementType>
ImageView<ElementType>::ImageView(const ImageView& other)
    : ImageBase<ElementType>(other.rows_, other.cols_, other.stride_bytes(),
                             other.num_elements_per_pixel_, other.data_) {}

template <typename ElementType>
ImageView<ElementType> ImageView<ElementType>::cropped(
    const ImageBoundingBox& bbox) const {
  NVBLOX_DCHECK(!bbox.isEmpty(), "Undefined bounding box not allowed");

  NVBLOX_DCHECK(bbox.min().y() >= 0 && bbox.max().y() < this->rows_,
                "Out of bounds");
  NVBLOX_DCHECK(bbox.min().x() >= 0 && bbox.max().x() < this->cols_,
                "Out of bounds");

  const int cropped_rows = bbox.max().y() - bbox.min().y() + 1;
  const int cropped_cols = bbox.max().x() - bbox.min().x() + 1;
  ElementType* cropped_data = this->data_ +
                              bbox.min().y() * this->stride_num_elements_ +
                              bbox.min().x() * this->num_elements_per_pixel_;
  return ImageView(cropped_rows, cropped_cols, this->stride_bytes(),
                   this->num_elements_per_pixel_, cropped_data);
}

template <typename ElementType>
ImageView<ElementType>& ImageView<ElementType>::operator=(
    const ImageView<ElementType>& other) {
  this->rows_ = other.rows_;
  this->cols_ = other.cols_;
  this->stride_num_elements_ = other.stride_num_elements_;
  this->num_elements_per_pixel_ = other.num_elements_per_pixel_;
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
  this->stride_num_elements_ = other.stride_num_elements_;
  this->num_elements_per_pixel_ = other.num_elements_per_pixel_;
  this->data_ = other.data_;
  return *this;
}

template <typename ElementType>
ImageView<ElementType>::ImageView(Image<ElementType>& image)
    : ImageView(image.rows(), image.cols(), image.dataPtr()) {}

template <typename ElementType>
ImageView<ElementType>::ImageView(const Image<ElementType_nonconst>& image)
    : ImageView(image.rows(), image.cols(), image.dataConstPtr()) {}

template <typename ElementType>
template <typename ImageType>
MaskedImageView<ElementType>::MaskedImageView(
    ImageType& image, const std::optional<ImageView<const uint8_t>>& mask,
    const MaskMode mode)
    : ImageView<ElementType>(image), mode_(mode) {
  static_assert(std::is_same<typename std::remove_cv<ElementType>::type,
                             typename ImageType::ElementType_nonconst>::value,
                "Image types must match");
  if (mask.has_value()) {
    NVBLOX_CHECK(mask.value().rows() == image.rows(),
                 "mask/image size mismatch");
    NVBLOX_CHECK(mask.value().cols() == image.cols(),
                 "mask/image size mismatch");
    mask_ = mask.value();
  }
}

template <typename ElementType>
bool MaskedImageView<ElementType>::isMasked(const int row_idx,
                                            const int col_idx) const {
  return
      // Always return "true" if no mask is given
      (mask_.dataConstPtr() == nullptr) ||
      // Non-inverted case
      (mode_ == MaskMode::kNonInverted) && mask_(row_idx, col_idx) ||
      // Inverted case
      (mode_ == MaskMode::kInverted) && !mask_(row_idx, col_idx);
}

}  // namespace nvblox
