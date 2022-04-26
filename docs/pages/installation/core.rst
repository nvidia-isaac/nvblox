=========================
Core Library Installation
=========================

There are two ways to install the nvblox core library: :ref:`natively <Native Installation>` on your system or inside a :ref:`docker container <Docker Installation>`.

Native Installation
===================

If you want to build natively, please follow these instructions. Instructions for docker are :ref:`further below <Docker>`.

Install dependencies
--------------------

We depend on:

* gtest
* glog
* gflags
* CUDA 10.2 - 11.5 (others might work but are untested)
* Eigen (no need to explicitly install, a recent version is built into the library)

Please run::

    sudo apt-get install -y libgoogle-glog-dev libgtest-dev libgflags-dev python3-dev
    cd /usr/src/googletest && sudo cmake . && sudo cmake --build . --target install

Build and run tests
-------------------
Build and run with::

    cd nvblox/nvblox
    mkdir build
    cd build
    cmake .. && make && ctest

All tests should pass.

Now you can run :ref:`core library example <Core Library Example - Native>`


Docker Installation
===================

We have several dockerfiles, each of which layers on top of the preceding one for the following purposes:

* **Docker.deps**

  - This sets up the environment and installs our dependencies.
  - This is used in our CI, where the later steps (building and testing) are taken care of by Jenkins (and not docker).
* **Docker.build**

  - Layers on top of Docker.deps.
  - This builds our package.
  - This is where you get off the layer train if you wanna run stuff (and don't care if it's tested).

* **Docker.test**

  - Layers on top of Docker.build.
  - Runs ours tests.
  - Useful for checking, on your machine, if things are likely to pass the tests in CI.

Install NVIDIA Container Toolkit
--------------------------------

We are reliant on nvidia docker. Install the `NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ following the instructions on that website.

We use the GPU during build, not only at run time. In the default configuration the GPU is only used at at runtime. One must therefore set the default runtime. Add `"default-runtime": "nvidia"` to `/etc/docker/daemon.json` such that it looks like::

    {
        "runtimes": {
            "nvidia": {
                "path": "/usr/bin/nvidia-container-runtime",
                "runtimeArgs": []
            } 
        },
        "default-runtime": "nvidia" 
    }

Restart docker::

    sudo systemctl restart docker

Build the Image
---------------

Now Let's build Dockerfile.deps docker image. This image install contains our dependencies. ::

    docker build -t nvblox_deps -f Dockerfile.deps .

Now let's build the Dockerfile.build. This image layers on the last, and actually builds the nvblox library. ::

    docker build -t nvblox -f Dockerfile.build .

Now you can run :ref:`core library example <Core Library Example - Docker>`


Additional instructions for Jetson Xavier
=========================================

These instructions are for a **native** build on the Jetson Xavier. A Docker based build is coming soon.

The instructions for the native build above work, with one exception:

We build using CMake's modern CUDA integration and therefore require a more modern version of CMAKE than (currently) ships with jetpack. Luckily the cmake developer team provide a means obtaining recent versions of CMake through apt.

1. Obtain a copy of the signing key::

    wget -qO - https://apt.kitware.com/keys/kitware-archive-latest.asc |
    sudo apt-key add -

2. Add the repository to your sources list::

    sudo apt-add-repository 'deb https://apt.kitware.com/ubuntu/ bionic main'
    sudo apt-get update

3. Update::

    sudo apt-get install cmake
