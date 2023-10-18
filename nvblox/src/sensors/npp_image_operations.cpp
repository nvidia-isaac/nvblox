/*
Copyright 2023 NVIDIA CORPORATION

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

#include <cuda_runtime.h>

#include "nvblox/core/cuda_stream.h"
#include "nvblox/core/internal/error_check.h"
#include "nvblox/sensors/image.h"
#include "nvblox/sensors/npp_image_operations.h"

namespace nvblox {
namespace image {

void printNPPVersionInfo() {
  const NppLibraryVersion* lib_ver_ptr = nppGetLibVersion();
  CHECK_NOTNULL(lib_ver_ptr);
  printf("NPP Library Version %d.%d.%d\n", lib_ver_ptr->major,
         lib_ver_ptr->minor, lib_ver_ptr->build);
}

NppStreamContext getNppStreamContext(const CudaStream& cuda_stream) {
  NppStreamContext context;
  context.hStream = cuda_stream.get();
  checkCudaErrors(cudaGetDevice(&context.nCudaDeviceId));
  cudaDeviceProp prop;
  checkCudaErrors(cudaGetDeviceProperties(&prop, context.nCudaDeviceId));
  context.nMultiProcessorCount = prop.multiProcessorCount;
  context.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
  context.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
  context.nSharedMemPerBlock = prop.sharedMemPerBlock;
  checkCudaErrors(cudaDeviceGetAttribute(
      &context.nCudaDevAttrComputeCapabilityMajor,
      cudaDevAttrComputeCapabilityMajor, context.nCudaDeviceId));
  checkCudaErrors(cudaDeviceGetAttribute(
      &context.nCudaDevAttrComputeCapabilityMinor,
      cudaDevAttrComputeCapabilityMinor, context.nCudaDeviceId));
  cudaStreamGetFlags(cuda_stream.get(), &context.nStreamFlags);
  return context;
}

void getInvalidDepthMaskAsync(const DepthImage& depth_image,
                              const NppStreamContext& npp_stream_context,
                              MonoImage* mask_ptr,
                              const float invalid_threshold) {
  CHECK_NOTNULL(mask_ptr);
  CHECK_EQ(depth_image.rows(), mask_ptr->rows());
  CHECK_EQ(depth_image.cols(), mask_ptr->cols());
  // ROI is the whole image
  const NppiSize roi_size{.width = depth_image.cols(),
                          .height = depth_image.rows()};

  checkNppErrors(nppiCompareC_32f_C1R_Ctx(
      depth_image.dataConstPtr(),          // pSrc
      depth_image.cols() * sizeof(float),  // nSrcStep
      invalid_threshold,                   // nConstant
      mask_ptr->dataPtr(),                 // pDst
      mask_ptr->cols() * sizeof(uint8_t),  // nDstStep
      roi_size,                            // oSizeROI
      NPP_CMP_LESS,                        // eComparisonOperation
      npp_stream_context                   // nppStreamCtx
      ));
}

void dilateMask3x3Async(const MonoImage& mask_image,
                        const NppStreamContext& npp_stream_context,
                        MonoImage* mask_dilated_ptr) {
  CHECK_NOTNULL(mask_dilated_ptr);
  CHECK_EQ(mask_image.rows(), mask_dilated_ptr->rows());
  CHECK_EQ(mask_image.cols(), mask_dilated_ptr->cols());
  const NppiSize roi_size{.width = mask_image.cols(),
                          .height = mask_image.rows()};
  const NppiSize size{.width = mask_image.cols(), .height = mask_image.rows()};
  const NppiPoint offset{.x = 0, .y = 0};
  checkNppErrors(nppiDilate3x3Border_8u_C1R_Ctx(
      mask_image.dataConstPtr(),                   // pSrc
      mask_image.cols() * sizeof(uint8_t),         // nSrcStep
      size,                                        // oSrcSize
      offset,                                      // oSrcOffset
      mask_dilated_ptr->dataPtr(),                 // pDst,
      mask_dilated_ptr->cols() * sizeof(uint8_t),  // nDstStep
      roi_size,                                    // oSizeROI
      NPP_BORDER_REPLICATE,                        // eBorderType
      npp_stream_context                           // nppStreamCtx
      ));
}

void maskedSetAsync(const MonoImage& mask, const float value,
                    const NppStreamContext& npp_stream_context,
                    DepthImage* depth_image_ptr) {
  CHECK_NOTNULL(depth_image_ptr);
  CHECK_EQ(depth_image_ptr->rows(), mask.rows());
  CHECK_EQ(depth_image_ptr->cols(), mask.cols());
  // Work on the whole image
  const NppiSize roi_size{.width = depth_image_ptr->cols(),
                          .height = depth_image_ptr->rows()};
  checkNppErrors(nppiSet_32f_C1MR_Ctx(
      value,                                    // nValue,
      depth_image_ptr->dataPtr(),               // pDst,
      depth_image_ptr->cols() * sizeof(float),  // nDstStep,
      roi_size,                                 // oSizeROI,
      mask.dataConstPtr(),                      // pMask,
      mask.cols() * sizeof(uint8_t),            // nMaskStep,
      npp_stream_context                        // nppStreamCtx
      ));
}

void setGreaterThanThresholdToValue(const MonoImage& image,
                                    const uint8_t threshold,
                                    const uint8_t value,
                                    const NppStreamContext& npp_stream_context,
                                    MonoImage* image_thresholded) {
  CHECK_NOTNULL(image_thresholded);
  CHECK_EQ(image.rows(), image_thresholded->rows());
  CHECK_EQ(image.cols(), image_thresholded->cols());

  const NppiSize roi_size{.width = image.cols(), .height = image.rows()};

  checkNppErrors(nppiThreshold_Val_8u_C1R_Ctx(
      image.dataConstPtr(),                         // pSrc
      image.cols() * sizeof(uint8_t),               // nSrcStep
      image_thresholded->dataPtr(),                 // pDst
      image_thresholded->cols() * sizeof(uint8_t),  // nDstStep
      roi_size,                                     // oSizeROI,
      threshold,                                    // nThreshold
      value,                                        // nValue
      NPP_CMP_GREATER,                              // eComparisonOperation,
      npp_stream_context                            // nppStreamCtx
      ));
}
}  // namespace image
}  // namespace nvblox
