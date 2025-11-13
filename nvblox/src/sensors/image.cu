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
#include "nvblox/utils/cuda_kernel_utils.h"

namespace nvblox {
namespace image {

template <typename ElementType>
ElementType maxGPUTemplate(const Image<ElementType>& image,
                           const CudaStream& cuda_stream) {
  const thrust::device_ptr<const ElementType> dev_ptr(image.dataConstPtr());
  const thrust::device_ptr<const ElementType> max_elem =
      thrust::max_element(thrust::device.on(cuda_stream), dev_ptr,
                          dev_ptr + (image.rows() * image.cols()));
  cuda_stream.synchronize();
  return *max_elem;
}

template <typename ElementType>
ElementType minGPUTemplate(const Image<ElementType>& image,
                           const CudaStream& cuda_stream) {
  const thrust::device_ptr<const ElementType> dev_ptr(image.dataConstPtr());
  const thrust::device_ptr<const ElementType> min_elem =
      thrust::min_element(thrust::device.on(cuda_stream), dev_ptr,
                          dev_ptr + (image.rows() * image.cols()));
  cuda_stream.synchronize();
  return *min_elem;
}

float maxGPU(const DepthImage& image, const CudaStream& cuda_stream) {
  return maxGPUTemplate(image, cuda_stream);
}

float minGPU(const DepthImage& image, const CudaStream& cuda_stream) {
  return minGPUTemplate(image, cuda_stream);
}

uint8_t maxGPU(const MonoImage& image, const CudaStream& cuda_stream) {
  return maxGPUTemplate(image, cuda_stream);
}

uint8_t minGPU(const MonoImage& image, const CudaStream& cuda_stream) {
  return minGPUTemplate(image, cuda_stream);
}

void minmaxGPU(const DepthImageConstView& image, float* min, float* max,
               const CudaStream& cuda_stream) {
  // Wrap our memory and reduce using thrust
  const thrust::device_ptr<const float> dev_ptr(image.dataConstPtr());
  const auto minmax_elem =
      thrust::minmax_element(thrust::device.on(cuda_stream), dev_ptr,
                             dev_ptr + (image.rows() * image.cols()));
  cuda_stream.synchronize();
  *min = *minmax_elem.first;
  *max = *minmax_elem.second;
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

void elementWiseMinInPlaceGPUAsync(const float constant, DepthImage* image,
                                   const CudaStream& cuda_stream) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device.on(cuda_stream), dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    min_with_constant_functor(constant));
}

void elementWiseMaxInPlaceGPUAsync(const float constant, DepthImage* image,
                                   const CudaStream& cuda_stream) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device.on(cuda_stream), dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    max_with_constant_functor(constant));
}

template <typename ImageType, typename OpType>
void elementWiseOpInPlaceGPUTemplateAsync(const ImageType& image_1,
                                          ImageType* image_2, OpType op,
                                          const CudaStream& cuda_stream) {
  using ElementType = typename ImageType::ElementType;
  CHECK_NOTNULL(image_2);
  CHECK_EQ(image_1.rows(), image_2->rows());
  CHECK_EQ(image_1.cols(), image_2->cols());
  thrust::device_ptr<const ElementType> dev_1_ptr(image_1.dataConstPtr());
  thrust::device_ptr<ElementType> dev_2_ptr(image_2->dataPtr());
  thrust::transform(thrust::device.on(cuda_stream), dev_1_ptr,
                    dev_1_ptr + image_1.numel(), dev_2_ptr, dev_2_ptr, op);
}

template <typename ImageType>
void elementWiseMaxInPlaceGPUTemplateAsync(const ImageType& image_1,
                                           ImageType* image_2,
                                           const CudaStream& cuda_stream) {
  using ElementType = typename ImageType::ElementType;
  elementWiseOpInPlaceGPUTemplateAsync(
      image_1, image_2, thrust::maximum<ElementType>(), cuda_stream);
}

void elementWiseMaxInPlaceGPUAsync(const DepthImage& image_1,
                                   DepthImage* image_2,
                                   const CudaStream& cuda_stream) {
  elementWiseMaxInPlaceGPUTemplateAsync(image_1, image_2, cuda_stream);
}

void elementWiseMaxInPlaceGPUAsync(const MonoImage& image_1, MonoImage* image_2,
                                   const CudaStream& cuda_stream) {
  elementWiseMaxInPlaceGPUTemplateAsync(image_1, image_2, cuda_stream);
}

template <typename ImageType>
void elementWiseMinInPlaceGPUTemplateAsync(const ImageType& image_1,
                                           ImageType* image_2,
                                           const CudaStream& cuda_stream) {
  using ElementType = typename ImageType::ElementType;
  elementWiseOpInPlaceGPUTemplateAsync(
      image_1, image_2, thrust::minimum<ElementType>(), cuda_stream);
}

void elementWiseMinInPlaceGPUAsync(const DepthImage& image_1,
                                   DepthImage* image_2,
                                   const CudaStream& cuda_stream) {
  elementWiseMinInPlaceGPUTemplateAsync(image_1, image_2, cuda_stream);
}

void elementWiseMinInPlaceGPUAsync(const MonoImage& image_1, MonoImage* image_2,
                                   const CudaStream& cuda_stream) {
  elementWiseMinInPlaceGPUTemplateAsync(image_1, image_2, cuda_stream);
}

struct multiply_with_constant_functor {
  const float constant_;

  multiply_with_constant_functor(float constant) : constant_(constant) {}

  __device__ float operator()(const float& pixel_value) const {
    return constant_ * pixel_value;
  }
};

void elementWiseMultiplicationInPlaceGPUAsync(const float constant,
                                              DepthImage* image,
                                              const CudaStream& cuda_stream) {
  thrust::device_ptr<float> dev_ptr(image->dataPtr());
  thrust::transform(thrust::device.on(cuda_stream), dev_ptr,
                    dev_ptr + (image->rows() * image->cols()), dev_ptr,
                    multiply_with_constant_functor(constant));
}

__device__ Color diff(const Color& color_1, const Color& color_2) {
  return Color(
      static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.r()) -
                                    static_cast<int16_t>(color_2.r()))),
      static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.g()) -
                                    static_cast<int16_t>(color_2.g()))),
      static_cast<uint8_t>(std::abs(static_cast<int16_t>(color_1.b()) -
                                    static_cast<int16_t>(color_2.b()))));
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
void getDifferenceImageGPUTemplateAsync(const ImageType& image_1,
                                        const ImageType& image_2,
                                        ImageType* diff_image_ptr,
                                        const CudaStream& cuda_stream) {
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
  dim3 gridShape(
      divideRoundUp(image_1.rows(), kThreadsPerBlockInEachDimension),
      divideRoundUp(image_1.cols(), kThreadsPerBlockInEachDimension));
  differenceImageKernel<<<gridShape, blockShape, 0, cuda_stream>>>(
      diff_image_ptr->dataPtr(), image_1.rows(), image_1.cols(),
      image_1.dataConstPtr(), image_2.dataConstPtr());
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
}

void getDifferenceImageGPUAsync(const DepthImage& image_1,
                                const DepthImage& image_2,
                                DepthImage* diff_image_ptr,
                                const CudaStream& cuda_stream) {
  getDifferenceImageGPUTemplateAsync(image_1, image_2, diff_image_ptr,
                                     cuda_stream);
}

void getDifferenceImageGPUAsync(const ColorImage& image_1,
                                const ColorImage& image_2,
                                ColorImage* diff_image_ptr,
                                const CudaStream& cuda_stream) {
  getDifferenceImageGPUTemplateAsync(image_1, image_2, diff_image_ptr,
                                     cuda_stream);
}

void getDifferenceImageGPUAsync(const MonoImage& image_1,
                                const MonoImage& image_2,
                                MonoImage* diff_image_ptr,
                                const CudaStream& cuda_stream) {
  getDifferenceImageGPUTemplateAsync(image_1, image_2, diff_image_ptr,
                                     cuda_stream);
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
void castTemplateAsync(const InputImageType& image_in,
                       OutputImageType* image_out_ptr,
                       const CudaStream& cuda_stream) {
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
  thrust::transform(thrust::device.on(cuda_stream), dev_input_ptr,
                    dev_input_ptr + (image_in.rows() * image_in.cols()),
                    dev_output_ptr,
                    cast_functor<OutputElementType, InputElementType>());
}

// Copy every second pixel from image_in to image_out
//
// @param image_in Input image
// @param image_out Output image. Must be half the size of image_in (rounding
// downwards)
//  n_threads: <= image_out.cols()
//  n_blocks: = image_out.rows()
__global__ void naiveDownscaleKernel(MonoImageConstView image_in,
                                     const int factor,
                                     MonoImageView image_out) {
  assert(image_out.rows() == image_in.rows() / factor);
  assert(image_out.cols() == image_in.cols() / factor);

  const int row = blockIdx.x;
  const int col_start = threadIdx.x;

  if (row < image_out.rows() && col_start < image_out.cols()) {
    for (int col = col_start; col < image_out.cols(); col += blockDim.x) {
      image_out(row, col) = image_in(row * factor, col * factor);
    }
  }
}

void naiveDownscaleGPUAsync(const MonoImage& image_in, const int factor,
                            MonoImage* image_out,
                            const CudaStream& cuda_stream) {
  const int new_rows = image_in.rows() / factor;
  const int new_cols = image_in.cols() / factor;

  // Only non-strided data supported
  CHECK_EQ(image_in.stride_num_elements(), image_in.cols());
  CHECK_EQ(image_out->stride_num_elements(), image_out->cols());
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  image_out->resizeAsync(new_rows, new_cols, cuda_stream);

  constexpr int kMaxNumThreadsPerBlock = 1024;
  const int num_blocks = new_rows;
  const int num_threads_per_block =
      std::min(kMaxNumThreadsPerBlock, image_out->cols());

  naiveDownscaleKernel<<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
      image_in, factor, *image_out);
}

__global__ void upscaleKernel(MonoImageConstView image_in, const int factor,
                              MonoImageView image_out) {
  assert(image_out.rows() == image_in.rows() * factor);
  assert(image_out.cols() == image_in.cols() * factor);

  const int row = blockIdx.x;
  const int col_start = threadIdx.x;

  if (row < image_out.rows() && col_start < image_out.cols()) {
    for (int col = col_start; col < image_out.cols(); col += blockDim.x) {
      image_out(row, col) = image_in(row / factor, col / factor);
    }
  }
}

void upscaleGPUAsync(const MonoImage& image_in, const int factor,
                     MonoImage* image_out, const CudaStream& cuda_stream) {
  const int new_rows = image_in.rows() * factor;
  const int new_cols = image_in.cols() * factor;

  // Only non-strided data supported
  CHECK_EQ(image_in.stride_num_elements(), image_in.cols());
  CHECK_EQ(image_out->stride_num_elements(), image_out->cols());
  CHECK_GT(new_rows, 0);
  CHECK_GT(new_cols, 0);

  image_out->resizeAsync(new_rows, new_cols, cuda_stream);

  constexpr int kMaxNumThreadsPerBlock = 1024;
  const int num_blocks = new_rows;
  const int num_threads_per_block =
      std::min(kMaxNumThreadsPerBlock, image_out->cols());

  upscaleKernel<<<num_blocks, num_threads_per_block, 0, cuda_stream>>>(
      image_in, factor, *image_out);
}

void castGPUAsync(const DepthImage& image_in, MonoImage* image_out_ptr,
                  const CudaStream& cuda_stream) {
  castTemplateAsync(image_in, image_out_ptr, cuda_stream);
}

}  // namespace image
}  // namespace nvblox
