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
#include <gtest/gtest.h>

#include "nvblox/sensors/camera.h"
#include "nvblox/sensors/lidar.h"
#include "nvblox/sensors/type_indexed_store.h"

namespace nvblox {

Camera createGtCamera() { return Camera(10.F, 11.F, 12.F, 13.F, 14, 15); }

void checkCamera(const Camera* camera) {
  ASSERT_NE(camera, nullptr);

  Camera gt = createGtCamera();
  EXPECT_EQ(camera->fu(), gt.fu());
  EXPECT_EQ(camera->fv(), gt.fv());
  EXPECT_EQ(camera->cu(), gt.cu());
  EXPECT_EQ(camera->cv(), gt.cv());
  EXPECT_EQ(camera->width(), gt.width());
  EXPECT_EQ(camera->height(), gt.height());
}

Lidar createGtLidar() { return Lidar(10, 11, 12.F, 14.F, 15.F); }

void checkLidar(const Lidar* lidar) {
  ASSERT_NE(lidar, nullptr);

  Lidar gt = createGtLidar();
  EXPECT_EQ(lidar->num_azimuth_divisions(), gt.num_azimuth_divisions());
  EXPECT_EQ(lidar->num_elevation_divisions(), gt.num_elevation_divisions());
  EXPECT_EQ(lidar->min_valid_range_m(), gt.min_valid_range_m());
  EXPECT_EQ(lidar->vertical_fov_rad(), gt.vertical_fov_rad());
}

TEST(TypeIndexedStoreTest, insertSingleCamera) {
  TypeIndexedStore store;

  ASSERT_FALSE(store.hasType<Camera>());
  store.set(createGtCamera());
  ASSERT_TRUE(store.hasType<Camera>());
  checkCamera(store.getPtr<Camera>());
  checkCamera(&store.get<Camera>());
}

TEST(TypeIndexedStoreTest, insertSingleLidar) {
  TypeIndexedStore store;

  ASSERT_FALSE(store.hasType<Lidar>());
  store.set(createGtLidar());
  ASSERT_TRUE(store.hasType<Lidar>());
  checkLidar(store.getPtr<Lidar>());
  checkLidar(&store.get<Lidar>());
}

TEST(TypeIndexedStoreTest, insertMulti) {
  TypeIndexedStore store;

  ASSERT_FALSE(store.hasType<Camera>());
  ASSERT_FALSE(store.hasType<Lidar>());
  store.set(createGtCamera());

  ASSERT_TRUE(store.hasType<Camera>());
  ASSERT_FALSE(store.hasType<Lidar>());
  store.set(createGtLidar());

  ASSERT_TRUE(store.hasType<Camera>());
  ASSERT_TRUE(store.hasType<Lidar>());

  checkCamera(store.getPtr<Camera>());
  checkCamera(&store.get<Camera>());
  checkLidar(store.getPtr<Lidar>());
  checkLidar(&store.get<Lidar>());
}

}  // namespace nvblox

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
