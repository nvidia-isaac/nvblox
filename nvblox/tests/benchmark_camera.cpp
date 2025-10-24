/*
Copyright 2025 NVIDIA CORPORATION

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
#include "nvblox/sensors/camera.h"

namespace nvblox {

constexpr static float fu_ = 300;
constexpr static float fv_ = 300;
constexpr static int width_ = 640;
constexpr static int height_ = 480;
constexpr static float cu_ = static_cast<float>(width_) / 2.0f;
constexpr static float cv_ = static_cast<float>(height_) / 2.0f;
constexpr static float k1_ = 0.1f;
constexpr static float k2_ = 0.1f;
constexpr static float k3_ = 0.01f;
constexpr static float p1_ = 0.01f;
constexpr static float p2_ = 0.02f;

Camera getTestCameraWithDistortion() {
  return Camera(fu_, fv_, cu_, cv_, width_, height_, k1_, k2_, k3_, p1_, p2_);
}

Camera getTestCameraWithoutDistortion() {
  return Camera(fu_, fv_, cu_, cv_, width_, height_);
}

// ------------------------------------------------------------------------
// Benchmarking functions for projecting points with and without distortion
// ------------------------------------------------------------------------
Vector2f projectPoints(const Camera& camera) {
  static constexpr int kNumIterations = 100000;

  Vector2f u_projection;
  for (int i = 0; i < kNumIterations; i++) {
    float x = i % 3 + 1;
    float y = i % 3 + 1;
    float z = i % 10 + 5;

    const Vector3f p_C{x, y, z};

    camera.project(p_C, &u_projection);
  }
  // Need to return the result to avoid the compiler optimizing away the
  // computation
  return u_projection;
}

void benchmarkProjectPoints_Distortion(benchmark::State& state) {
  const Camera camera = getTestCameraWithDistortion();

  for (auto _ : state) {
    Vector2f u_projection = projectPoints(camera);
    benchmark::DoNotOptimize(u_projection);
  }
}
BENCHMARK(benchmarkProjectPoints_Distortion)->Unit(benchmark::kMillisecond);

void benchmarkProjectPoints_NoDistortion(benchmark::State& state) {
  const Camera camera = getTestCameraWithoutDistortion();

  for (auto _ : state) {
    Vector2f u_projection = projectPoints(camera);
    benchmark::DoNotOptimize(u_projection);
  }
}
BENCHMARK(benchmarkProjectPoints_NoDistortion)->Unit(benchmark::kMillisecond);

// ------------------------------------------------------------------------
// Benchmarking functions for unprojecting points with and without distortion
// ------------------------------------------------------------------------

Vector3f unprojectPoints(const Camera& camera) {
  Vector3f p_C;
  for (int x = 0; x < camera.width(); x++) {
    for (int y = 0; y < camera.height(); y++) {
      Vector2f u_px(x, y);
      p_C = camera.unprojectFromImagePlaneCoordinates(u_px, 10.0);
    }
  }
  return p_C;
}

void benchmarkUnprojectPoints_Distortion(benchmark::State& state) {
  const Camera camera = getTestCameraWithDistortion();

  for (auto _ : state) {
    Vector3f p_C = unprojectPoints(camera);
    benchmark::DoNotOptimize(p_C);
  }
}
BENCHMARK(benchmarkUnprojectPoints_Distortion)->Unit(benchmark::kMillisecond);

void benchmarkUnprojectPoints_NoDistortion(benchmark::State& state) {
  const Camera camera = getTestCameraWithoutDistortion();

  for (auto _ : state) {
    Vector3f p_C = unprojectPoints(camera);
    benchmark::DoNotOptimize(p_C);
  }
}
BENCHMARK(benchmarkUnprojectPoints_NoDistortion)->Unit(benchmark::kMillisecond);

}  // namespace nvblox

BENCHMARK_MAIN();
