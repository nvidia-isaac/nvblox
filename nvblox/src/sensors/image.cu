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
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/transform.h>

#include <algorithm>

#include "nvblox/sensors/image.h"

namespace nvblox {
namespace image {

template <typename ElementType>
ElementType maxGPUTemplate(const Image<ElementType>& image) {
  const thrust::device_ptr<const ElementType> dev_ptr(image.dataConstPtr());
  const thrust::device_ptr<const ElementType> max_elem = thrust::max_element(
      thrust::device, dev_ptr, dev_ptr + (image.rows() * image.cols()));
  return *max_elem;
}

template <typename ElementType>
ElementType minGPUTemplate(const Image<ElementType>& image) {
  const thrust::device_ptr<const ElementType> dev_ptr(image.dataConstPtr());
  const thrust::device_ptr<const ElementType> min_elem = thrust::min_element(
      thrust::device, dev_ptr, dev_ptr + (image.rows() * image.cols()));
  return *min_elem;
}

float maxGPU(const DepthImage& image) { return maxGPUTemplate(image); }

float minGPU(const DepthImage& image) { return minGPUTemplate(image); }

uint8_t maxGPU(const MonoImage& image) { return maxGPUTemplate(image); }

uint8_t minGPU(const MonoImage& image) { return minGPUTemplate(image); }

std::pair<float, float> minmaxGPU(const DepthImage& image) {
  // Wrap our memory and reduce using thrust
  const thrust::device_ptr<const float> dev_ptr(image.dataConstPtr());
  const auto minmax_elem = thrust::minmax_element(
      thrust::device, dev_ptr, dev_ptr + (image.rows() * image.cols()));
  return {*minmax_elem.first, *minmax_elem.second};
}

struct max_with_constant_functor {
  const float constant_;

  max_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return fmaxf(pixel_value, constant_);
  }
};

struct min_with_constant_functor {
  const float constant_;

  min_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return fminf(pixel_value, constant_);
  }
};

void elementWiseMinInPlaceGPU(const float constant, DepthImage* image) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device, dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    min_with_constant_functor(constant));
}

void elementWiseMaxInPlaceGPU(const float constant, DepthImage* image) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device, dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    max_with_constant_functor(constant));
}

template <typename ImageType, typename OpType>
void elementWiseOpInPlaceGPUTemplate(const ImageType& image_1,
                                     ImageType* image_2, OpType op) {
  using ElementType = typename ImageType::ElementType;
  CHECK_NOTNULL(image_2);
  CHECK_EQ(image_1.rows(), image_2->rows());
  CHECK_EQ(image_1.cols(), image_2->cols());
  thrust::device_ptr<const ElementType> dev_1_ptr(image_1.dataConstPtr());
  thrust::device_ptr<ElementType> dev_2_ptr(image_2->dataPtr());
  thrust::transform(thrust::device, dev_1_ptr, dev_1_ptr + image_1.numel(),
                    dev_2_ptr, dev_2_ptr, op);
}

template <typename ImageType>
void elementWiseMaxInPlaceGPUTemplate(const ImageType& image_1,
                                      ImageType* image_2) {
  using ElementType = typename ImageType::ElementType;
  elementWiseOpInPlaceGPUTemplate(image_1, image_2,
                                  thrust::maximum<ElementType>());
}

void elementWiseMaxInPlaceGPU(const DepthImage& image_1, DepthImage* image_2) {
  elementWiseMaxInPlaceGPUTemplate(image_1, image_2);
}

void elementWiseMaxInPlaceGPU(const MonoImage& image_1, MonoImage* image_2) {
  elementWiseMaxInPlaceGPUTemplate(image_1, image_2);
}

template <typename ImageType>
void elementWiseMinInPlaceGPUTemplate(const ImageType& image_1,
                                      ImageType* image_2) {
  using ElementType = typename ImageType::ElementType;
  elementWiseOpInPlaceGPUTemplate(image_1, image_2,
                                  thrust::minimum<ElementType>());
}

void elementWiseMinInPlaceGPU(const DepthImage& image_1, DepthImage* image_2) {
  elementWiseMinInPlaceGPUTemplate(image_1, image_2);
}

void elementWiseMinInPlaceGPU(const MonoImage& image_1, MonoImage* image_2) {
  elementWiseMinInPlaceGPUTemplate(image_1, image_2);
}

struct multiply_with_constant_functor {
  const float constant_;

  multiply_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return constant_ * pixel_value;
  }
};

void elementWiseMultiplicationInPlaceGPU(const float constant,
                                         DepthImage* image) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device, dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    multiply_with_constant_functor(constant));
}

__device__ Color diff(const Color& color_1, const Color& color_2) {
  return Color(static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.r) -
                                             static_cast<int16_t>(color_2.r))),
               static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.g) -
                                             static_cast<int16_t>(color_2.g))),
               static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.b) -
                                             static_cast<int16_t>(color_2.b))),
               static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.a) -
                                             static_cast<int16_t>(color_2.a))));
}

__device__ float diff(const float& depth_1, const float& depth_2) {
  return fabsf(depth_1 - depth_2);
}

__device__ uint8_t diff(const uint8_t& uint_1, const uint8_t& uint_2) {
  return abs(uint_1 - uint_2);
}

template <typename ElementType>
__global__ void differenceImageKernel(ElementType* diff_image_ptr,
                                      const int rows, const int cols,
                                      const ElementType* image_1,
                                      const ElementType* image_2) {
  // NOTE(alexmillane): Memory access is fully coallesed because neighbouring
  // threads in the grid x dimension, access neighbouring memory elements
  // (row-major images).
  const int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int col_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (col_idx < cols && row_idx < rows) {
    const ElementType color_1 = image::access(row_idx, col_idx, cols, image_1);
    const ElementType color_2 = image::access(row_idx, col_idx, cols, image_2);
    const ElementType abs_color_diff = diff(color_1, color_2);
    image::access(row_idx, col_idx, cols, diff_image_ptr) = abs_color_diff;
  }
}

template <typename ImageType>
void getDifferenceImageGPUTemplate(const ImageType& image_1,
                                   const ImageType& image_2,
                                   ImageType* diff_image_ptr) {
  CHECK_NOTNULL(diff_image_ptr);
  CHECK_EQ(image_1.rows(), image_2.rows());
  CHECK_EQ(image_1.cols(), image_2.cols());
  CHECK(image_1.memory_type() == MemoryType::kDevice ||
        image_1.memory_type() == MemoryType::kUnified);
  CHECK(image_2.memory_type() == MemoryType::kDevice ||
        image_2.memory_type() == MemoryType::kUnified);
  // If output is the wrong size, reallocate
  if (diff_image_ptr->rows() != image_1.rows() ||
      diff_image_ptr->cols() != image_1.cols()) {
    LOG(INFO) << "Allocating output image for image difference.";
    *diff_image_ptr =
        ImageType(image_1.rows(), image_1.cols(), image_1.memory_type());
  }
  CHECK_EQ(image_1.rows(), diff_image_ptr->rows());
  CHECK_EQ(image_1.cols(), diff_image_ptr->cols());
  // One thread per pixel (lol)
  constexpr int kThreadsPerBlockInEachDimension = 8;
  dim3 blockShape(kThreadsPerBlockInEachDimension,
                  kThreadsPerBlockInEachDimension);
  dim3 gridShape((image_1.rows() / kThreadsPerBlockInEachDimension) + 1,
                 (image_1.cols() / kThreadsPerBlockInEachDimension) + 1);
  differenceImageKernel<<<gridShape, blockShape>>>(
      diff_image_ptr->dataPtr(), image_1.rows(), image_1.cols(),
      image_1.dataConstPtr(), image_2.dataConstPtr());
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void getDifferenceImageGPU(const DepthImage& image_1, const DepthImage& image_2,
                           DepthImage* diff_image_ptr) {
  getDifferenceImageGPUTemplate(image_1, image_2, diff_image_ptr);
}

void getDifferenceImageGPU(const ColorImage& image_1, const ColorImage& image_2,
                           ColorImage* diff_image_ptr) {
  getDifferenceImageGPUTemplate(image_1, image_2, diff_image_ptr);
}

void getDifferenceImageGPU(const MonoImage& image_1, const MonoImage& image_2,
                           MonoImage* diff_image_ptr) {
  getDifferenceImageGPUTemplate(image_1, image_2, diff_image_ptr);
}

template <typename OutputType, typename InputType>
struct cast_functor {
  cast_functor() {}

  __device__ OutputType operator()(const InputType& pixel_value) const {
    return pixel_value;
  }
};

template <typename OutputType, typename InputType>
__device__ OutputType elementCast(const InputType input) {
  return static_cast<OutputType>(input);
}

template <typename InputImageType, typename OutputImageType>
void castTemplate(const InputImageType& image_in,
                  OutputImageType* image_out_ptr) {
  CHECK(image_in.memory_type() == MemoryType::kDevice ||
        image_in.memory_type() == MemoryType::kUnified);
  if (image_in.rows() != image_out_ptr->rows() ||
      image_in.cols() != image_out_ptr->cols()) {
    LOG(INFO) << "Allocating output image";
    *image_out_ptr = OutputImageType(image_in.rows(), image_in.cols(),
                                     image_in.memory_type());
  }

  using OutputElementType = typename OutputImageType::ElementType;
  using InputElementType = typename InputImageType::ElementType;

  thrust::device_ptr<const InputElementType> dev_input_ptr(
      image_in.dataConstPtr());
  thrust::device_ptr<OutputElementType> dev_output_ptr(
      image_out_ptr->dataPtr());
  thrust::transform(thrust::device, dev_input_ptr,
                    dev_input_ptr + (image_in.rows() * image_in.cols()),
                    dev_output_ptr,
                    cast_functor<OutputElementType, InputElementType>());
}

void castGPU(const DepthImage& image_in, MonoImage* image_out_ptr) {
  castTemplate(image_in, image_out_ptr);
}

}  // namespace image
}  // namespace nvblox
