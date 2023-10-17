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
#pragma once

#include "cuda_runtime.h"

namespace nvblox {

/// A thin interface to a CudaStream

/// @brief A thin interface to a CudaStream
/// This class is an interface class and cannot be created directly. Instead
/// create one of the derived classes which implements owning or non-owning
/// semantics.
class CudaStream {
 public:
  virtual ~CudaStream() = default;

  // Copying streams interface
  CudaStream(const CudaStream& other) = default;
  CudaStream(CudaStream&& other) = default;
  CudaStream& operator=(const CudaStream& other) = default;
  CudaStream& operator=(CudaStream&& other) = default;

  /// Returns the underlying CUDA stream
  /// @return The raw CUDA stream
  cudaStream_t& get() { return *stream_ptr_; }
  const cudaStream_t& get() const { return *stream_ptr_; }

  operator cudaStream_t() { return *stream_ptr_; }
  operator cudaStream_t() const { return *stream_ptr_; }

  /// Synchronize the stream
  void synchronize() const;

 protected:
  CudaStream(cudaStream_t* stream_ptr) : stream_ptr_(stream_ptr) {}

  cudaStream_t* stream_ptr_;
};

/// @brief A simple RAII holder for a cuda stream.
/// This class setups up a stream on construction and cleans up on destruction.
/// The destructor synchronizes the stream prior to cleaning up.
class CudaStreamOwning : public CudaStream {
 public:
  /// Creates the stream on construction and synchronizes+destroys on
  /// destruction.
  CudaStreamOwning();
  virtual ~CudaStreamOwning();

  // Can't copy owning streams (because both copies would want ownership)
  // NOTE(alexmillane): We *could* implement move operations if that becomes
  // important. For now streams are also un-movable.
  CudaStreamOwning(const CudaStreamOwning& other) = delete;
  CudaStreamOwning(CudaStreamOwning&& other) = delete;

  // Can't assign streams
  // NOTE(alexmillane): We *could* implement move operations if that becomes
  // important. For now streams are also un-movable.
  CudaStreamOwning& operator=(const CudaStreamOwning& other) = delete;
  CudaStreamOwning& operator=(CudaStreamOwning&& other) = delete;

 protected:
  cudaStream_t stream_;
};

/// @brief A thin wrapping around an already initialized stream.
/// Note that this class should be used as a last resort, usually to interface
/// with streams passed outside nvblox. It is up to the user to ensure that the
/// underlying stream is valid for the duration of the lifetime of this
/// instances of this class.
class CudaStreamNonOwning : public CudaStream {
 public:
  CudaStreamNonOwning() = delete;
  CudaStreamNonOwning(cudaStream_t* stream) : CudaStream(stream) {}

  virtual ~CudaStreamNonOwning() = default;

  // Can copy non-owning streams
  CudaStreamNonOwning(const CudaStreamNonOwning& other) = default;
  CudaStreamNonOwning(CudaStreamNonOwning&& other) = default;
  CudaStreamNonOwning& operator=(const CudaStreamNonOwning& other) = default;
  CudaStreamNonOwning& operator=(CudaStreamNonOwning&& other) = default;
};

}  // namespace nvblox
