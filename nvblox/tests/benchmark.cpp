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

#include <benchmark/benchmark.h>
#include <gtest/gtest.h>
#include <cassert>
#include <memory>
#include "nvblox/datasets/3dmatch.h"
#include "nvblox/executables/fuser.h"
#include "nvblox/io/image_io.h"
#include "nvblox/primitives/scene.h"
#include "nvblox/sensors/mask_preprocessor.h"
#include "nvblox/sensors/npp_image_operations.h"
#include "nvblox/serialization/mesh_serializer_gpu.h"
#include "nvblox/tests/integrator_utils.h"
#include "nvblox/tests/utils.h"

namespace nvblox {

// Create a mapper with suitable default  parameters
std::unique_ptr<Mapper> createMapper() {
  constexpr float kVoxelSize = 0.05;
  auto mapper = std::make_unique<Mapper>(kVoxelSize, MemoryType::kDevice,
                                         ProjectiveLayerType::kTsdf);

  // Default parameters
  mapper->color_integrator().max_integration_distance_m(5.0f);
  mapper->tsdf_integrator().max_integration_distance_m(5.0f);
  mapper->tsdf_integrator().view_calculator().raycast_subsampling_factor(4);
  mapper->occupancy_integrator().max_integration_distance_m(5.0f);
  mapper->occupancy_integrator().view_calculator().raycast_subsampling_factor(
      4);
  mapper->esdf_integrator().max_esdf_distance_m(4.0f);
  mapper->esdf_integrator().min_weight(2.0f);

  return mapper;
}

// Prevent GLOG from being initialized more than once
std::once_flag init_glog_flag;

struct FrameData final {
  DepthImage depth_frame{MemoryType::kHost};
  Transform T_L_C;
  Camera camera;
  ColorImage color_frame{MemoryType::kHost};
};

FrameData readFrameData() {
  constexpr bool kUseMultithreaded = false;
  auto data_loader = datasets::threedmatch::DataLoader::create(
      "../tests/data/3dmatch", 1, kUseMultithreaded);

  FrameData data;
  if (data_loader->loadNext(&data.depth_frame, &data.T_L_C, &data.camera,
                            &data.color_frame) !=
      datasets::DataLoadResult::kSuccess) {
    LOG(ERROR)
        << "Invalid dataset path. Hint: Run benchmarking from the build dir";
    std::abort();
  }
  return data;
}

void benchmarkAll(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();

  for (auto _ : state) {
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
    mapper->integrateColor(data.color_frame, data.T_L_C, data.camera);
    mapper->updateColorMesh();
    mapper->updateEsdf();
  }
}
BENCHMARK(benchmarkAll)->Unit(benchmark::kMillisecond);
BENCHMARK(benchmarkAll)->Unit(benchmark::kMillisecond)->Iterations(100);

void benchmarkIntegrateDepth(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();

  for (auto _ : state) {
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
  }
}
BENCHMARK(benchmarkIntegrateDepth)->Unit(benchmark::kMillisecond);

void benchmarkIntegrateColor(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();

  for (auto _ : state) {
    state.PauseTiming();
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
    state.ResumeTiming();

    mapper->integrateColor(data.color_frame, data.T_L_C, data.camera);
  }
}
BENCHMARK(benchmarkIntegrateColor)->Unit(benchmark::kMillisecond);

void benchmarkUpdateColorMesh(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();

  for (auto _ : state) {
    state.PauseTiming();
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
    mapper->integrateColor(data.color_frame, data.T_L_C, data.camera);
    state.ResumeTiming();

    mapper->updateColorMesh();
  }
}
BENCHMARK(benchmarkUpdateColorMesh)->Unit(benchmark::kMillisecond);

void benchmarkUpdateEsdf(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();

  for (auto _ : state) {
    state.PauseTiming();
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
    mapper->integrateColor(data.color_frame, data.T_L_C, data.camera);
    state.ResumeTiming();

    mapper->updateEsdf();
  }
}
BENCHMARK(benchmarkUpdateEsdf)->Unit(benchmark::kMillisecond);

void benchmarkSerializeMesh(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });
  const FrameData data = readFrameData();
  auto mapper = createMapper();
  ColorMeshSerializerGpu serializer;

  host_vector<Vector3f> vertices;
  host_vector<Color> colors;
  host_vector<int> triangle_indices;

  CudaStreamOwning cuda_stream;

  for (auto _ : state) {
    state.PauseTiming();
    mapper->integrateDepth(data.depth_frame, data.T_L_C, data.camera);
    mapper->integrateColor(data.color_frame, data.T_L_C, data.camera);
    mapper->updateColorMesh();
    state.ResumeTiming();

    serializer.serialize(mapper->color_mesh_layer(),
                         mapper->color_mesh_layer().getAllBlockIndices(),
                         cuda_stream);
  }
}

BENCHMARK(benchmarkSerializeMesh)->Unit(benchmark::kMillisecond);

void benchmarkRemoveSmallConnectedComponents(benchmark::State& state) {
  std::call_once(init_glog_flag, []() { google::InitGoogleLogging(""); });

  MonoImage mask(MemoryType::kDevice);
  createMaskImage(&mask,
                  static_cast<test_utils::MaskImageType>(state.range(0)));
  MonoImage mask_out(mask.rows(), mask.cols(), MemoryType::kDevice);
  image::MaskPreprocessor mask_preprocessor(
      std::make_shared<CudaStreamOwning>());

  for (auto _ : state) {
    mask_preprocessor.removeSmallConnectedComponents(mask, 10000, &mask_out);
  }
}
BENCHMARK(benchmarkRemoveSmallConnectedComponents)
    ->Unit(benchmark::kMillisecond)
    ->Arg(static_cast<int64_t>(test_utils::MaskImageType::kFromDisk))
    ->Arg(static_cast<int64_t>(test_utils::MaskImageType::kEverythingZero))
    ->Arg(static_cast<int64_t>(test_utils::MaskImageType::kEverythingFilled))
    ->Arg(static_cast<int64_t>(test_utils::MaskImageType::kGrid))
    ->Arg(static_cast<int64_t>(test_utils::MaskImageType::kTwoSquares));

void benchmarkMonoImageGpuToCpuRoundtrip(benchmark::State& state) {
  const int32_t width = state.range(0);
  const int32_t height = state.range(1);

  MonoImage image_host(height, width, MemoryType::kHost);
  MonoImage image_device(height, width, MemoryType::kDevice);
  image_host.setZeroAsync(CudaStreamOwning());
  image_device.setZeroAsync(CudaStreamOwning());

  for (auto _ : state) {
    image_host.copyFrom(image_device);
    image_device.copyFrom(image_host);
  }
}
BENCHMARK(benchmarkMonoImageGpuToCpuRoundtrip)
    ->Args({320, 200})
    ->Args({640, 480})
    ->Args({1024, 640})
    ->Args({1920, 1080})
    ->Unit(benchmark::kMillisecond);

void benchmarkFreespaceUpdate(benchmark::State& state) {
  primitives::Scene scene = test_utils::getSphereInBox();
  constexpr float kVoxelSizeM = 0.05;
  constexpr int kMaxDistVox = 4;
  constexpr float kMaxDistM = static_cast<float>(kMaxDistVox) * kVoxelSizeM;
  TsdfLayer tsdf_layer(kVoxelSizeM, MemoryType::kUnified);
  scene.generateLayerFromScene(kMaxDistM, &tsdf_layer);

  auto cuda_stream = std::make_shared<CudaStreamOwning>();
  FreespaceIntegrator freespace_integrator(cuda_stream);
  freespace_integrator.check_neighborhood(state.range(0));
  LOG(INFO) << "Check neighborhood: "
            << freespace_integrator.check_neighborhood();
  FreespaceLayer freespace_layer(kVoxelSizeM, MemoryType::kDevice);
  const auto all_blocks = tsdf_layer.getAllBlockIndices();

  nvblox::Time time(1);
  // Let's first run a warmup round to avoid measuring the intial memory
  // reset/transfer
  freespace_integrator.updateFreespaceLayer<Camera>(
      all_blocks, time, tsdf_layer, {}, &freespace_layer);

  for (auto _ : state) {
    time += Time(100000);

    freespace_integrator.updateFreespaceLayer<Camera>(
        all_blocks, time, tsdf_layer, {}, &freespace_layer);
  }
}

BENCHMARK(benchmarkFreespaceUpdate)
    ->Arg(false)
    ->Arg(true)
    ->Unit(benchmark::kMillisecond);

}  // namespace nvblox

BENCHMARK_MAIN();
