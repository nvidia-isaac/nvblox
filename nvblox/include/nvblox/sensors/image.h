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
#include <memory>
#include <optional>

#include "nvblox/core/color.h"
#include "nvblox/core/feature_array.h"
#include "nvblox/core/types.h"
#include "nvblox/core/unified_vector.h"

namespace nvblox {

/// Row-major access to images (on host and device)
namespace image {

// The value of a pixel in a mask image which is "masked".
static constexpr uint8_t kMaskedValue = 255;

template <typename ElementType>
__host__ __device__ inline const ElementType& access(int row_idx, int col_idx,
                                                     int element_index,
                                                     int stride_num_elements,
                                                     int num_elements_per_pixel,
                                                     const ElementType* data) {
  return data[row_idx * stride_num_elements + col_idx * num_elements_per_pixel +
              element_index];
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int row_idx, int col_idx,
                                               int element_index,
                                               int stride_num_elements,
                                               int num_elements_per_pixel,
                                               ElementType* data) {
  return data[row_idx * stride_num_elements + col_idx * num_elements_per_pixel +
              element_index];
}

template <typename ElementType>
__host__ __device__ inline const ElementType& access(int row_idx, int col_idx,
                                                     int stride_num_elements,
                                                     const ElementType* data) {
  constexpr int kElementIndex = 0;
  constexpr int kNumElementsPerPixel = 1;
  return access(row_idx, col_idx, kElementIndex, stride_num_elements,
                kNumElementsPerPixel, data);
}

template <typename ElementType>
__host__ __device__ inline ElementType& access(int row_idx, int col_idx,
                                               int stride_num_elements,
                                               ElementType* data) {
  constexpr int kElementIndex = 0;
  constexpr int kNumElementsPerPixel = 1;
  return access(row_idx, col_idx, kElementIndex, stride_num_elements,
                kNumElementsPerPixel, data);
}

template <typename ElementType>
__host__ __device__ inline const ElementType& access(int linear_idx,
                                                     const ElementType* data) {
  return data[linear_idx];
}
template <typename ElementType>
__host__ __device__ inline ElementType& access(int linear_idx,
                                               ElementType* data) {
  return data[linear_idx];
}

}  // namespace image

constexpr MemoryType kDefaultImageMemoryType = MemoryType::kDevice;

template <typename _ElementType>
class ImageBase {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;
  typedef typename std::remove_cv<_ElementType>::type ElementType_nonconst;

  /// Destructor
  __host__ __device__ virtual ~ImageBase() = default;

  /// Attributes
  __host__ __device__ inline int cols() const { return cols_; }
  __host__ __device__ inline int rows() const { return rows_; }
  __host__ __device__ inline int stride_bytes() const {
    return stride_num_elements_ * sizeof(ElementType);
  }
  __host__ __device__ inline int stride_num_elements() const {
    return stride_num_elements_;
  }

  __host__ __device__ inline int num_elements_per_pixel() const {
    return num_elements_per_pixel_;
  }

  __host__ __device__ inline int numel() const {
    return cols_ * rows_ * num_elements_per_pixel_;
  }
  __host__ __device__ inline int width() const { return cols_; }
  __host__ __device__ inline int height() const { return rows_; }

  /// Access
  /// NOTE(alexmillane): The guard-rails are off here. If you declare a kDevice
  /// Image and try to access its data, you will get undefined behaviour.
  __host__ __device__ inline const ElementType& operator()(
      const int row_idx, const int col_idx) const {
    NVBLOX_DCHECK(row_idx < rows_ && row_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(col_idx < cols_ && col_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(num_elements_per_pixel_ == 1,
                  "Use dedicated accessor when num_elements_per_pixel > 1");
    return image::access(row_idx, col_idx, stride_num_elements_, data_);
  }
  __host__ __device__ inline ElementType& operator()(const int row_idx,
                                                     const int col_idx) {
    NVBLOX_DCHECK(row_idx < rows_, "Requested pixel out of bounds");
    NVBLOX_DCHECK(col_idx < cols_, "Requested pixel out of bounds");
    NVBLOX_DCHECK(num_elements_per_pixel_ == 1,
                  "Use dedicated accessor when num_elements_per_pixel > 1");
    return image::access(row_idx, col_idx, stride_num_elements_, data_);
  }

  __host__ __device__ inline const ElementType& operator()(
      const int row_idx, const int col_idx, const int element_idx) const {
    NVBLOX_DCHECK(row_idx < rows_ && row_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(col_idx < cols_ && col_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(element_idx < num_elements_per_pixel_,
                  "Requested element out of bounds");
    return image::access(row_idx, col_idx, element_idx, stride_num_elements_,
                         num_elements_per_pixel_, data_);
  }

  __host__ __device__ inline ElementType& operator()(const int row_idx,
                                                     const int col_idx,
                                                     const int element_idx) {
    NVBLOX_DCHECK(row_idx < rows_ && row_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(col_idx < cols_ && col_idx >= 0,
                  "Requested pixel out of bounds");
    NVBLOX_DCHECK(element_idx < num_elements_per_pixel_,
                  "Requested element out of bounds");
    return image::access(row_idx, col_idx, element_idx, stride_num_elements_,
                         num_elements_per_pixel_, data_);
  }

  __host__ __device__ inline const ElementType& operator()(
      const int linear_idx) const {
    NVBLOX_DCHECK(linear_idx < numel(), "Linear index out of bounds");
    NVBLOX_DCHECK(stride_num_elements_ == cols_,
                  "Linear indexing requires non-padded image");
    return image::access(linear_idx, data_);
  }
  __host__ __device__ inline ElementType& operator()(const int linear_idx) {
    NVBLOX_DCHECK(linear_idx < numel(), "Linear index out of bounds");
    NVBLOX_DCHECK(stride_num_elements_ == cols_,
                  "Linear indexing requires non-padded image");
    return image::access(linear_idx, data_);
  }

  /// Raw pointer access
  __host__ __device__ inline ElementType* dataPtr() { return data_; }
  __host__ __device__ inline const ElementType* dataConstPtr() const {
    return data_;
  }

  __host__ __device__ static int strideFromBytesToElements(
      const int stride_bytes) {
    NVBLOX_DCHECK(stride_bytes % (sizeof(ElementType)) == 0,
                  "Stride must be a multiple of the element size in bytes");
    return stride_bytes / (sizeof(ElementType));
  }

 protected:
  /// Constructors protected. Only callable from the child classes.
  template <typename OtherElementType = ElementType>
  __host__ __device__ ImageBase(int rows, int cols, int stride_bytes,
                                int num_elements_per_pixel,
                                OtherElementType* data = nullptr)
      : rows_(rows),
        cols_(cols),
        stride_num_elements_(strideFromBytesToElements(stride_bytes)),
        num_elements_per_pixel_(num_elements_per_pixel),
        data_(data) {}
  template <typename OtherElementType = ElementType>
  __host__ __device__ ImageBase(int rows, int cols, int stride_bytes,
                                OtherElementType* data = nullptr)
      : rows_(rows),
        cols_(cols),
        stride_num_elements_(strideFromBytesToElements(stride_bytes)),
        num_elements_per_pixel_(1),
        data_(data) {}
  template <typename OtherElementType = ElementType>
  __host__ __device__ ImageBase(int rows, int cols,
                                OtherElementType* data = nullptr)
      : rows_(rows),
        cols_(cols),
        stride_num_elements_(cols),
        num_elements_per_pixel_(1),
        data_(data) {}
  __host__ __device__ ImageBase()
      : rows_(0),
        cols_(0),
        stride_num_elements_(0),
        num_elements_per_pixel_(1),
        data_(nullptr) {}

  // Delete copying
  ImageBase(const ImageBase& other) = delete;
  ImageBase& operator=(const ImageBase& other) = delete;

  // Delete moving
  ImageBase(ImageBase&& other) = delete;
  ImageBase& operator=(ImageBase&& other) = delete;

  // Height of the image
  int rows_ = 0;
  // Width of the image
  int cols_ = 0;
  // Number of elements between the same x-coordinate in two successive rows.
  int stride_num_elements_ = 0;
  // How many ElementTypes are stored in each xy coordinate?
  int num_elements_per_pixel_ = 1;
  // Image data
  ElementType* data_ = nullptr;
};

/// Row-major image.
/// - Note that in a row-major image, rows follow one another in linear memory,
/// which means the column index varied between subsequent elements.
/// - Images use corner based indexing such that the pixel with index (0,0) is
/// centered at (0.5px,0.5px) and spans (0.0px,0.0px) to (1.0px,1.0px).
/// - Points on the image plane are defined as: u_px = (u_px.x(), u_px.y()) =
/// (col_idx, row_idx) in pixels.
template <typename _ElementType>
class Image : public ImageBase<_ElementType> {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;

  Image() = delete;
  explicit Image(MemoryType memory_type)
      : memory_type_(memory_type), owned_data_(0, memory_type) {}

  Image(int rows, int cols, MemoryType memory_type = kDefaultImageMemoryType);

  __host__ __device__ virtual ~Image() = default;

  /// Move constructor and assignment
  Image(Image&& other);
  Image& operator=(Image&& other);

  // Attributes
  inline MemoryType memory_type() const { return memory_type_; }

  /// Set the image to 0.
  void setZeroAsync(const CudaStream& cuda_stream);

  /// Deep copy from other image
  void copyFrom(const ImageBase<ElementType>& other);
  void copyFromAsync(const ImageBase<ElementType>& other,
                     const CudaStream& cuda_stream);
  void copyFromAsync(const ImageBase<const ElementType>& other,
                     const CudaStream& cuda_stream);

  /// Copy from other buffer and reallocate if necessary
  void copyFromAsync(const size_t rows, const size_t cols,
                     const ElementType* const buffer,
                     const CudaStream& cuda_stream);
  void copyFrom(const size_t rows, const size_t cols,
                const ElementType* const buffer);
  void copyFromAsync(const size_t rows, const size_t cols,
                     const size_t stride_num_elements,
                     const size_t num_elements_per_pixel,
                     const ElementType* const buffer,
                     const CudaStream& cuda_stream);
  void copyFrom(const size_t rows, const size_t cols,
                const size_t stride_num_elements,
                const size_t num_elements_per_pixel,
                const ElementType* const buffer);

  // Copy to a buffer. We assume the buffer has sufficient capacity.
  void copyToAsync(ElementType* buffer, const CudaStream& cuda_stream) const;
  void copyTo(ElementType* buffer) const;

  // Resize the image an re-allocate the owned buffer if needed
  void resizeAsync(const size_t rows, const size_t cols,
                   const CudaStream& cuda_stream);

 protected:
  // TODO(dtingdahl) remove memory_type member and instead rely on the one in
  // owned_data
  MemoryType memory_type_;
  unified_vector<ElementType> owned_data_;
};

/// ImageView that wraps an existing image buffer
///
///  * Can wrap externally allocated image into nvblox without having to copy
///    the whole buffer.
///  * Lightweight: Can be constructed on the fly without much overhead.
///  * Can be used inside CUDA kernels for safe pixel access (no need to pass
///    raw pointers)
///  * Does not control memory lifetime. It is the users responsibility to
///    ensure that the underlying memory exists throughout use.
template <typename _ElementType>
class ImageView : public ImageBase<_ElementType> {
 public:
  // "Save" the element type so it's queryable as Image::ElementType.
  typedef _ElementType ElementType;
  typedef typename std::remove_cv<_ElementType>::type ElementType_nonconst;

  __host__ __device__ ImageView() = default;

  /// Wrap an externally allocated data buffer as an image view.
  ///
  /// @param rows rows in image buffer
  /// @param cols cols in image buffer
  /// @param data memory buffer
  template <typename OtherElementType = ElementType>
  __host__ __device__ ImageView(int rows, int cols,
                                OtherElementType* data = nullptr);

  /// Wrap an externally allocated data buffer as an image view. This version
  /// supports more options for defining the layout of the wrapped buffer.
  ///
  /// @param rows rows in image buffer
  /// @param cols cols in image buffer
  /// @param stride_bytes Buffer stride bwetween rows *in bytes*
  /// @param Number of elements stored in each pixel.
  /// @param data memory buffer
  template <typename OtherElementType = ElementType>
  __host__ __device__ ImageView(int rows, int cols, int stride_bytes,
                                int num_elements_per_pixel,
                                OtherElementType* data = nullptr);

  /// Construct from an (memory-owning) Image
  /// Note that it is the users responsiblity to ensure the underlying image
  /// outlives the constructed view.
  /// @param image The (memory-owning) Image
  ImageView(Image<ElementType>& image);
  ImageView(const Image<ElementType_nonconst>& image);

  /// Destructor
  __host__ __device__ virtual ~ImageView() = default;

  /// (Shallow) Copy
  __host__ __device__ ImageView(const ImageView& other);
  __host__ __device__ ImageView& operator=(const ImageView& other);

  /// Move (ImageView is a shallow copy so a move-construction is the same as
  /// copy-construction)
  __host__ __device__ ImageView(ImageView&& other);
  __host__ __device__ ImageView& operator=(ImageView&& other);

  /// Return a cropped version of the image view
  __host__ __device__ ImageView cropped(const ImageBoundingBox& bbox) const;
};

/// An ImageView with an accompanying view of a boolean mask.
/// If no mask is provided during construction, the mask will have value "true"
/// everywhere. The "mode" argument can be used to optionally invert the
/// provided mask
enum class MaskMode { kNonInverted, kInverted };

// This constant can be passed instead of a mask in order to mimic a mask that
// is active everywhere
constexpr std::nullopt_t kMaskActiveEverywhere = std::nullopt;
template <typename _ElementType>
class MaskedImageView : public ImageView<_ElementType> {
 public:
  typedef _ElementType ElementType;
  typedef typename std::remove_cv<_ElementType>::type ElementType_nonconst;

  MaskedImageView() = default;

  /// Copy-construct from other masked image view. Templated on OtherElementType
  /// to support const and non-const
  template <typename OtherElementType>
  MaskedImageView(const MaskedImageView<OtherElementType>& other)
      : ImageView<ElementType>(other),
        mask_(other.mask()),
        mode_(other.mode()) {}

  /// Assign-construct from other masked image view. Templated on
  /// OtherElementType to support const and non-const
  template <typename OtherElementType>
  MaskedImageView operator=(const MaskedImageView<OtherElementType>& other) {
    return MaskedImageView(other);
  }

  /// Return true if an explicit mask image was provided during construction.
  bool hasMask() const { return mask_.dataConstPtr() != nullptr; }

  /// Construct from existing image and mask. Templated on OtherImageType to
  /// support const and non-const versions of ImageView and Image. Construct
  /// from existing image and mask view
  template <typename OtherImageType>
  MaskedImageView(OtherImageType& image,
                  const std::optional<ImageView<const uint8_t>>& mask,
                  const MaskMode mode = MaskMode::kNonInverted);

  MaskedImageView(const int rows, const int cols, ElementType* data);

  /// Probe whether a pixel is masked or not. Will always return true if no mask
  /// image was provided during construction.
  __device__ __host__ bool isMasked(const int row_idx, const int col_idx) const;

  /// Get the mask view
  ImageView<const uint8_t> mask() const { return mask_; }

  /// Get the mask mode
  MaskMode mode() const { return mode_; }

 private:
  ImageView<const uint8_t> mask_;
  MaskMode mode_ = MaskMode::kNonInverted;
};

/// Common Names
using DepthImage = Image<float>;
using ColorImage = Image<Color>;
using MonoImage = Image<uint8_t>;
using DepthImageView = ImageView<float>;
using ColorImageView = ImageView<Color>;
using MonoImageView = ImageView<uint8_t>;
using MaskedDepthImageView = MaskedImageView<float>;
using DepthImageConstView = ImageView<const float>;
using ColorImageConstView = ImageView<const Color>;
using MonoImageConstView = ImageView<const uint8_t>;
using MaskedDepthImageConstView = MaskedImageView<const float>;
using FeatureImage = Image<FeatureArray>;
using FeatureImageView = ImageView<FeatureArray>;
using FeatureImageConstView = ImageView<const FeatureArray>;
using MaskedFeatureImageConstView = MaskedImageView<const FeatureArray>;
using MaskedColorImageConstView = MaskedImageView<const Color>;

// Image Operations
namespace image {

// Note that the min/max operations synchronizes the cuda stream
float maxGPU(const DepthImage& image, const CudaStream& cuda_stream);
float minGPU(const DepthImage& image, const CudaStream& cuda_stream);
uint8_t maxGPU(const MonoImage& image, const CudaStream& cuda_stream);
uint8_t minGPU(const MonoImage& image, const CudaStream& cuda_stream);
void minmaxGPU(const DepthImageConstView& image, float* min, float* max,
               const CudaStream& cuda_stream);

void elementWiseMinInPlaceGPUAsync(const float constant, DepthImage* image,
                                   const CudaStream& cuda_stream);
void elementWiseMaxInPlaceGPUAsync(const float constant, DepthImage* image,
                                   const CudaStream& cuda_stream);

void elementWiseMaxInPlaceGPUAsync(const DepthImage& image_1,
                                   DepthImage* image_2,
                                   const CudaStream& cuda_stream);
void elementWiseMaxInPlaceGPUAsync(const MonoImage& image_1, MonoImage* image_2,
                                   const CudaStream& cuda_stream);
void elementWiseMinInPlaceGPUAsync(const DepthImage& image_1,
                                   DepthImage* image_2,
                                   const CudaStream& cuda_stream);
void elementWiseMinInPlaceGPUAsync(const MonoImage& image_1, MonoImage* image_2,
                                   const CudaStream& cuda_stream);

void elementWiseMultiplicationInPlaceGPUAsync(const float constant,
                                              DepthImage* image,
                                              const CudaStream& cuda_stream);

void getDifferenceImageGPUAsync(const DepthImage& image_1,
                                const DepthImage& image_2,
                                DepthImage* diff_image_ptr,
                                const CudaStream& cuda_stream);
void getDifferenceImageGPUAsync(const ColorImage& image_1,
                                const ColorImage& image_2,
                                ColorImage* diff_image_ptr,
                                const CudaStream& cuda_stream);
void getDifferenceImageGPUAsync(const MonoImage& image_1,
                                const MonoImage& image_2,
                                MonoImage* diff_image_ptr,
                                const CudaStream& cuda_stream);

void castGPUAsync(const DepthImage& image_in, MonoImage* image_out_ptr,
                  const CudaStream& cuda_stream);

// Downscale an image by an integer factor. Does not perform any kind of
// pre-processing so be prepared for aliasing effects.
void naiveDownscaleGPUAsync(const MonoImage& image_in, const int factor,
                            MonoImage* image_out,
                            const CudaStream& cuda_stream);

// Upscale an image by an integer factor.
void upscaleGPUAsync(const MonoImage& image_in, const int factor,
                     MonoImage* image_out, const CudaStream& cuda_stream);

}  // namespace image
}  // namespace nvblox

#include "nvblox/sensors/internal/impl/image_impl.h"
