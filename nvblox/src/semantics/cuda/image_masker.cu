#include "nvblox/semantics/image_masker.h"

#include "nvblox/core/cuda/atomic_float.cuh"

namespace nvblox {

template <typename ElementType>
__device__ inline ElementType getInvalidPixel();

template <>
__device__ inline float getInvalidPixel<float>() {
  return -1.0f;
}

template <>
__device__ inline Color getInvalidPixel<Color>() {
  return Color(0, 0, 0, 0);
}

template <typename ElementType>
__device__ inline void copyToUnmaskedOutput(const ElementType* input,
                                            const int row_idx,
                                            const int col_idx, const int cols,
                                            ElementType* unmasked_output,
                                            ElementType* masked_output) {
  image::access(row_idx, col_idx, cols, unmasked_output) =
      image::access(row_idx, col_idx, cols, input);
  image::access(row_idx, col_idx, cols, masked_output) =
      getInvalidPixel<ElementType>();
}

template <typename ElementType>
__device__ inline void copyToMaskedOutput(const ElementType* input,
                                          const int row_idx, const int col_idx,
                                          const int cols,
                                          ElementType* unmasked_output,
                                          ElementType* masked_output) {
  image::access(row_idx, col_idx, cols, unmasked_output) =
      getInvalidPixel<ElementType>();
  image::access(row_idx, col_idx, cols, masked_output) =
      image::access(row_idx, col_idx, cols, input);
}

template <typename ElementType>
__global__ void splitImageKernel(const ElementType* input, const uint8_t* mask,
                                 const int rows, const int cols,
                                 ElementType* unmasked_output,
                                 ElementType* masked_output) {
  // Each thread does a single pixel
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // Assuming that the mask lies directly on top of the input image
  const bool is_masked = image::access(row_idx, col_idx, cols, mask);
  if (is_masked) {
    copyToMaskedOutput(input, row_idx, col_idx, cols, unmasked_output,
                       masked_output);
  } else {
    copyToUnmaskedOutput(input, row_idx, col_idx, cols, unmasked_output,
                         masked_output);
  }
}

__global__ void initializeImageKernel(const float value, const int rows,
                                      const int cols, float* image) {
  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }
  image::access(row_idx, col_idx, cols, image) = value;
}

template <unsigned int kPatchSize>
__global__ void getMinimumDepthKernel(const float* depth_input,
                                      const Transform T_CM_CD,
                                      const Camera depth_camera,
                                      const Camera mask_camera, const int rows,
                                      const int cols, float* min_depth_image) {
  static_assert((kPatchSize % 2) == 1,
                "Patch size of getMinimumDepthKernel must be odd.");

  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // Unproject from the image.
  const Index2D u_CD(col_idx, row_idx);
  const float depth = image::access(row_idx, col_idx, cols, depth_input);
  const Vector3f p_CD = depth_camera.unprojectFromPixelIndices(u_CD, depth);

  // Transform from the depth camera to the mask camera frame
  const Vector3f p_CM = T_CM_CD * p_CD;

  // Project on the mask image
  Eigen::Vector2f u_CM;
  if (mask_camera.project(p_CM, &u_CM)) {
    // Update the minimum depth values inside the image patch
    // TODO: think about replacing the for loops with a more efficient
    //       implementation
    for (int patch_row = 0; patch_row < kPatchSize; patch_row++) {
      for (int patch_col = 0; patch_col < kPatchSize; patch_col++) {
        const int absolute_col = u_CM.x() + patch_col - kPatchSize / 2;
        const int absolute_row = u_CM.y() + patch_row - kPatchSize / 2;

        if ((absolute_row >= 0) && (absolute_row < mask_camera.rows()) &&
            (absolute_col >= 0) && (absolute_col < mask_camera.cols())) {
          // Update the minimal depth values seen from the mask camera
          atomicMinFloat(
              &image::access(absolute_row, absolute_col, cols, min_depth_image),
              p_CM.z());
        }
      }
    }
  }
}

__global__ void splitImageKernel(
    const float* depth_input, const uint8_t* mask, const Transform T_CM_CD,
    const Camera depth_camera, const Camera mask_camera,
    const float occlusion_threshold, const int rows, const int cols,
    float* unmasked_depth_output, float* masked_depth_output,
    const float* min_depth_image) {
  // Each thread does a single pixel on the depth input image
  const int col_idx = threadIdx.x + blockIdx.x * blockDim.x;
  const int row_idx = threadIdx.y + blockIdx.y * blockDim.y;
  if ((row_idx >= rows) || (col_idx >= cols)) {
    return;
  }

  // If the depth is infinite, the input pixel is not masked
  const float depth = image::access(row_idx, col_idx, cols, depth_input);
  if (std::isinf(depth)) {
    copyToUnmaskedOutput(depth_input, row_idx, col_idx, cols,
                         unmasked_depth_output, masked_depth_output);
    return;
  }

  // Unproject from the image.
  const Index2D u_CD(col_idx, row_idx);
  const Vector3f p_CD = depth_camera.unprojectFromPixelIndices(u_CD, depth);

  // Transform from the depth camera to the mask camera frame
  const Vector3f p_CM = T_CM_CD * p_CD;

  // Project on the mask image
  Eigen::Vector2f u_CM;
  if (mask_camera.project(p_CM, &u_CM)) {
    // A point is considered to be occluded on the mask image only if it lies
    // more than the occlusion threshold behind the point occluding it.
    const bool is_occluded =
        image::access(u_CM.y(), u_CM.x(), cols, min_depth_image) +
            occlusion_threshold <
        p_CM.z();
    const bool is_masked = image::access(u_CM.y(), u_CM.x(), cols, mask);

    // A masked point is only valid if it is not occluded on the mask image.
    if (is_masked && !is_occluded) {
      copyToMaskedOutput(depth_input, row_idx, col_idx, cols,
                         unmasked_depth_output, masked_depth_output);
    } else {
      copyToUnmaskedOutput(depth_input, row_idx, col_idx, cols,
                           unmasked_depth_output, masked_depth_output);
    }
  } else {
    // If the projection failed, the input pixel is not masked
    copyToUnmaskedOutput(depth_input, row_idx, col_idx, cols,
                         unmasked_depth_output, masked_depth_output);
  }
}

ImageMasker::ImageMasker() { checkCudaErrors(cudaStreamCreate(&cuda_stream_)); }

ImageMasker::~ImageMasker() {
  cudaStreamSynchronize(cuda_stream_);
  checkCudaErrors(cudaStreamDestroy(cuda_stream_));
}

float ImageMasker::occlusion_threshold() const {
  return occlusion_threshold_m_;
}

void ImageMasker::occlusion_threshold(float occlusion_threshold) {
  occlusion_threshold_m_ = occlusion_threshold;
}

void ImageMasker::splitImageOnGPU(const DepthImage& input,
                                  const MonoImage& mask,
                                  DepthImage* unmasked_output,
                                  DepthImage* masked_output) {
  splitImageOnGPUTemplate(input, mask, unmasked_output, masked_output);
}

void ImageMasker::splitImageOnGPU(const ColorImage& input,
                                  const MonoImage& mask,
                                  ColorImage* unmasked_output,
                                  ColorImage* masked_output) {
  splitImageOnGPUTemplate(input, mask, unmasked_output, masked_output);
}

void ImageMasker::splitImageOnGPU(const DepthImage& depth_input,
                                  const MonoImage& mask,
                                  const Transform& T_CM_CD,
                                  const Camera& depth_camera,
                                  const Camera& mask_camera,
                                  DepthImage* unmasked_depth_output,
                                  DepthImage* masked_depth_output) {
  timing::Timer image_masking_timer("image_masker/split_depth_image");
  allocateOutput(depth_input, unmasked_depth_output, masked_depth_output);

  // Kernel call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(depth_input.cols() / kThreadsPerThreadBlock.x + 1,
                        depth_input.rows() / kThreadsPerThreadBlock.y + 1, 1);

  // Initialize the minimum depth image
  constexpr float max_value = std::numeric_limits<float>::max();
  DepthImage min_depth_image =
      DepthImage(mask.rows(), mask.cols(), mask.memory_type());
  initializeImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                          cuda_stream_>>>(max_value,                   // NOLINT
                                          min_depth_image.rows(),      // NOLINT
                                          min_depth_image.cols(),      // NOLINT
                                          min_depth_image.dataPtr());  // NOLINT

  // Find the minimal depth values seen from the mask camera
  constexpr uint8_t kPatchSize = 5;
  getMinimumDepthKernel<kPatchSize>
      <<<num_blocks, kThreadsPerThreadBlock, 0, cuda_stream_>>>(
          depth_input.dataConstPtr(),  // NOLINT
          T_CM_CD,                     // NOLINT
          depth_camera,                // NOLINT
          mask_camera,                 // NOLINT
          depth_input.rows(),          // NOLINT
          depth_input.cols(),          // NOLINT
          min_depth_image.dataPtr());  // NOLINT

  // Split the depth image according to the mask considering occlusion.
  splitImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0, cuda_stream_>>>(
      depth_input.dataConstPtr(),        // NOLINT
      mask.dataConstPtr(),               // NOLINT
      T_CM_CD,                           // NOLINT
      depth_camera,                      // NOLINT
      mask_camera,                       // NOLINT
      occlusion_threshold_m_,            // NOLINT
      depth_input.rows(),                // NOLINT
      depth_input.cols(),                // NOLINT
      unmasked_depth_output->dataPtr(),  // NOLINT
      masked_depth_output->dataPtr(),    // NOLINT
      min_depth_image.dataConstPtr());   // NOLINT

  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
  image_masking_timer.Stop();
}

template <typename ImageType>
void ImageMasker::splitImageOnGPUTemplate(const ImageType& input,
                                          const MonoImage& mask,
                                          ImageType* unmasked_output,
                                          ImageType* masked_output) {
  allocateOutput(input, unmasked_output, masked_output);
  CHECK((input.rows() == mask.rows()) && (input.cols() == mask.cols()));

  // Kernel
  // Call params
  // - 1 thread per pixel
  // - 8 x 8 threads per thread block
  // - N x M thread blocks get 1 thread per pixel
  constexpr dim3 kThreadsPerThreadBlock(8, 8, 1);
  const dim3 num_blocks(input.cols() / kThreadsPerThreadBlock.x + 1,  // NOLINT
                        input.rows() / kThreadsPerThreadBlock.y + 1,  // NOLINT
                        1);
  splitImageKernel<<<num_blocks, kThreadsPerThreadBlock, 0,
                     cuda_stream_>>>(input.dataConstPtr(),        // NOLINT
                                     mask.dataConstPtr(),         // NOLINT
                                     input.rows(),                // NOLINT
                                     input.cols(),                // NOLINT
                                     unmasked_output->dataPtr(),  // NOLINT
                                     masked_output->dataPtr()     // NOLINT
  );
  checkCudaErrors(cudaStreamSynchronize(cuda_stream_));
  checkCudaErrors(cudaPeekAtLastError());
}

template <typename ImageType>
void ImageMasker::allocateOutput(const ImageType& input,
                                 ImageType* unmasked_output,
                                 ImageType* masked_output) {
  CHECK_NOTNULL(unmasked_output);
  CHECK_NOTNULL(masked_output);
  CHECK_GT(input.rows(), 0);
  CHECK_GT(input.cols(), 0);

  // Allocate output images if required
  if ((input.rows() != unmasked_output->rows()) ||
      (input.cols() != unmasked_output->cols())) {
    *unmasked_output =
        ImageType(input.rows(), input.cols(), input.memory_type());
  }
  if ((input.rows() != masked_output->rows()) ||
      (input.cols() != masked_output->cols())) {
    *masked_output = ImageType(input.rows(), input.cols(), input.memory_type());
  }
  CHECK((input.rows() == unmasked_output->rows()) &&
        (input.cols() == unmasked_output->cols()));
  CHECK((input.rows() == masked_output->rows()) &&
        (input.cols() == masked_output->cols()));
}

}  // namespace nvblox
