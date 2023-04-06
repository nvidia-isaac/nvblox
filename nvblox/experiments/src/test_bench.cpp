#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nvblox/nvblox.h>

#include "nvblox/datasets/3dmatch.h"

using namespace nvblox;

int main(int argc, char* argv[]) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_alsologtostderr = true;
  google::InstallFailureSignalHandler();

  /*
   * Uncomment if your experiment needs an image
   */

  // Get the dataset
  // std::string base_path;
  // if (argc < 2) {
  //   // Try out running on the test datasets.
  //   base_path = "../tests/data/3dmatch";
  //   LOG(WARNING) << "No 3DMatch file path specified; defaulting to the test "
  //                   "directory.";
  // } else {
  //   base_path = argv[1];
  //   LOG(INFO) << "Loading 3DMatch files from " << base_path;
  // }

  // constexpr int kSeqNum = 1;
  // constexpr bool kMultithreaded = false;
  // auto data_loader = std::make_unique<datasets::threedmatch::DataLoader>(
  //     base_path, kSeqNum, kMultithreaded);
  // if (!data_loader) {
  //   LOG(FATAL) << "Error creating the DataLoader";
  // }
  // DepthImage depth_image;
  // ColorImage color_image;
  // Transform T_L_C;
  // Camera camera;
  // data_loader->loadNext(&depth_image, &T_L_C, &camera, &color_image);

  LOG(INFO) << "Stub file, intended to be a playground for trying things.";
  LOG(INFO) << "Add your experimentation code here.";

  /*
   * Place your experiment code here.
   */

  return 0;
}
