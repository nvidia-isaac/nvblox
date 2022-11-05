# Develop nvblox
### Development
1. Nvblox offers API to integrate LiDAR frames, but users need to write the specific API to use the KITTI and FusionPortable dataset
    1. [std::move](https://en.cppreference.com/w/cpp/utility/move): an efficient way to move resources from A to B 

2. (solved) ROS1 interface: users need to create the nvblox_ros to use ROS

### Code Pipeline

1. Data loader

2. Frame integration

   1. After loading data (the code API): Fuser::integrateFrame(const int frame_number)

   2. RgbdMapper::integrateOSLidarDepth -> ProjectiveTsdfIntegrator::integrateFrame -> ProjectiveTsdfIntegrator::integrateFrameTemplate 

      > * set <code>voxel_size</code> and <code>truncation_distance</code>
      > * Identify blocks given the camera view: <code>view_calculator_.getBlocksInImageViewRaycast</code>
      >   * <code>getBlocksByRaycastingPixels</code>: Raycasts through (possibly subsampled) pixels in the image, use the kernal function
      >   * <code>*void* combinedBlockIndicesInImageKernel</code>: retrieve visiable block by raycasting voxels, done in GPU
      >
      >
      > * TSDF integration given block indices: <code>integrateBlocksTemplate</code>
      >
      >   * <code>ProjectiveTsdfIntegrator::integrateBlocks</code>: block integration for the OSLidar, use the kernal function
      >   * <code>integrateBlocksKernel</code>: TSDF integration for each block, done in GPU
      >     * <code>projectThreadVoxel</code>: convert blocks' indices into coordinates, retrieve voxels from the block, and project them onto the image to check whether they are visible or not
      >     * <code>interpolateOSLidarImage</code>: linear interpolation of depth images given float coordinates
      >     * <code>updateVoxel</code>: update the TSDF values of all visible voxels. 

### Run an example with the FusionPortable dataset

#### [Tricks to preprocess OSLiDAR points](docs/preprocess_OSLiDAR.md)

#### Demo 

Create a docker container

```
docker pull iidcramlab/nvblox:20221105-ros-noetic-open3d
xhost local:docker
docker run -it --net=host --env="DISPLAY" -v $HOME/.Xauthority:/root/.Xauthority:rw \
-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
--name nvblox \
iidcramlab/nvblox:20221105-ros-noetic-open3d /bin/bash
```

Download data

1. FusionPortable_20220216_garden_day: http://gofile.me/72EEc/Fa0eMVojB

Navigate to and run the `fuse_fusionportable` binary. From the nvblox base folder run

```
cd nvblox/build \
make \
./executables/fuse_fusionportable 20220216_garden_day/ \
tsdf_integrator_max_integration_distance_m 50 \
-num_frames 300 \
-voxel_size 0.1 \
-mesh_output_path 20220216_garden_day/mesh_0.1.ply
```

Once it's done we can view the output mesh using the Open3D viewer.

```
python3 ../../visualization/visualize_mesh.py 20220216_garden_day/mesh_0.1.ply
```

--------------------------
# nvblox
Signed Distance Functions (SDFs) on NVIDIA GPUs.

An SDF library which offers
* Support for storage of various voxel types
* GPU accelerated agorithms such as:
  * TSDF construction
  * ESDF construction
* ROS2 interface (see [isaac_ros_nvblox](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox))
* ~~Python bindings~~ (coming soon)

Do we need another SDF library? That depends on your use case. If you're interested in:
* **Path planning**: We provide GPU accelerated, incremental algorithms for calculating the Euclidian Signed Distance Field (ESDF) which is useful for colision checking and therefore robotic pathplanning. In contrast, existing GPU-accelerated libraries target reconstruction only, and are therefore generally not useful in a robotics context.
* **GPU acceleration**: Our previous works [voxblox](https://github.com/ethz-asl/voxblox) and [voxgraph](https://github.com/ethz-asl/voxgraph) are used for path planning, however utilize CPU compute only, which limits the speed of these toolboxes (and therefore the resolution of the maps they can build in real-time).

Here we show slices through a distance function generated from *nvblox* using data from the [3DMatch dataset](https://3dmatch.cs.princeton.edu/), specifically the [Sun3D](http://sun3d.cs.princeton.edu/) `mit_76_studyroom` scene:

<div align="center"><img src="docs/images/nvblox_slice.gif" width=600px/></div>

# Note from the authors
This package is under active development. Feel free to make an issue for bugs or feature requests, and we always welcome pull requests!

# ROS2 Interface
This repo contains the core library which can be linked into users' projects. If you want to use nvblox on a robot out-of-the-box, please see our [ROS2 interface](https://github.com/NVIDIA-ISAAC-ROS/isaac_ros_nvblox), which downloads and builds the core library during installation.

# Native Installation
If you want to build natively, please follow these instructions. Instructions for docker are [further below](#docker).

## Install dependencies
We depend on:
- gtest
- glog
- gflags (to run experiments)
- CUDA 10.2 - 11.5 (others might work but are untested)
- Eigen (no need to explicitly install, a recent version is built into the library)
- SQLite 3 (for serialization)
Please run
```
sudo apt-get install -y libgoogle-glog-dev libgtest-dev libgflags-dev python3-dev libsqlite3-dev
cd /usr/src/googletest && sudo cmake . && sudo cmake --build . --target install
```

## Build and run tests
```
cd nvblox/nvblox
mkdir build
cd build
cmake .. && make && cd tests && ctest
```

## Run an example

In this example we fuse data from the [3DMatch dataset](https://3dmatch.cs.princeton.edu/). First let's grab the dataset. Here I'm downloading it to my dataset folder `~/dataset/3dmatch`.
```
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P ~/datasets/3dmatch
unzip ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d ~/datasets/3dmatch
```
Navigate to and run the `fuse_3dmatch` binary. From the nvblox base folder run
```
cd nvblox/build/experiments
./fuse_3dmatch ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/ --esdf_frame_subsampling 3000 --mesh_output_path mesh.ply
```
Once it's done we can view the output mesh using the Open3D viewer.
```
pip3 install open3d
python3 ../../visualization/visualize_mesh.py mesh.ply
```
you should see a mesh of a room:
<div align="center"><img src="docs/images/reconstruction_in_docker_trim.png" width=600px/></div>

# Docker

We have several dockerfiles which layer on top of one another for the following purposes:

* **Docker.deps**
* * This installs our dependencies.
* * This is used in our CI, where the later steps (building and testing) are taken care of by Jenkins (and not docker).
* **Docker.build**
* * Layers on top of Docker.deps.
* * This builds our package.
* * This is where you get off the layer train if you wanna run stuff (and don't care if it's tested).
* **Docker.test**
* * Layers on top of Docker.build.
* * Runs ours tests.
* * Useful for checking if things are likely to pass the tests in CI.

We are reliant on nvidia docker. Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) following the instructions on that website.

We use the GPU during build, not only at run time. In the default configuration the GPU is only used at at runtime. One must therefore set the default runtime. Add `"default-runtime": "nvidia"` to `/etc/docker/daemon.json` such that it looks like:
```
{
    "runtimes": {
        "nvidia": {
            "path": "/usr/bin/nvidia-container-runtime",
            "runtimeArgs": []
         } 
    },
    "default-runtime": "nvidia" 
}
```
Restart docker
```
sudo systemctl restart docker
```
Now Let's build Dockerfile.deps docker image. This image install contains our dependacies. 
```
docker build -t nvblox_deps -f Dockerfile.deps .
```
Now let's build the Dockerfile.build. This image layers on the last, and actually builds the nvblox library.
```
docker build -t nvblox -f Dockerfile.build .
```
Now let's run the 3DMatch example inside the docker. Note there's some additional complexity in the `docker run` command such that we can forward X11 to the host (we're going to be view a reconstruction in a GUI). Run the container using:
```
xhost local:docker
docker run -it --net=host --env="DISPLAY" -v $HOME/.Xauthority:/root/.Xauthority:rw -v /tmp/.X11-unix:/tmp/.X11-unix:rw nvblox
```
Let's download a dataset and run the example (this is largely a repeat of "Run an example" above).
```
apt-get update
apt-get install unzip
wget http://vision.princeton.edu/projects/2016/3DMatch/downloads/rgbd-datasets/sun3d-mit_76_studyroom-76-1studyroom2.zip -P ~/datasets/3dmatch
unzip ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2.zip -d ~/datasets/3dmatch
cd nvblox/nvblox/build/experiments/
./fuse_3dmatch ~/datasets/3dmatch/sun3d-mit_76_studyroom-76-1studyroom2/ --esdf_frame_subsampling 3000 --mesh_output_path mesh.ply
```
Now let's visualize. From the same experiments folder run:
```
apt-get install python3-pip libgl1-mesa-glx
pip3 install open3d
python3 ../../visualization/visualize_mesh.py mesh.ply
```

# Additional instructions for Jetson Xavier
These instructions are for a native build on the Jetson Xavier. A Docker based build is coming soon.

The instructions for the native build above work, with one exception:

We build using CMake's modern CUDA integration and therefore require a more modern version of CMAKE than (currently) ships with jetpack. Luckily the Cmake developer team provide a means obtaining recent versions of CMake through apt.

1. Obtain a copy of the signing key:
```
wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc |
    sudo apt-key add -
```
2. Add the repository to your sources list and update.
```
sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
sudo apt-get update
```
3. Update!
```
sudo apt-get install cmake
```
4. To use the python examples, it is also necessary to make numpy play nice with the Jetson. You can do that by adding the following to your `~/.bashrc`:
```
export OPENBLAS_CORETYPE=ARMV8
```

# License
This code is under an [open-source license](LICENSE) (Apache 2.0). :)

# Reference
[1] Parallel Banding Algorithm to Compute Exact Distance Transform with the GPU
> compute EDT with the GPU