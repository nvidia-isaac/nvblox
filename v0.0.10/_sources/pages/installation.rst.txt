Installation
============

There are several ways to install ``nvblox``. See :ref:`supported_platforms_table` for a list of which methods are supported on which platforms.

1. :ref:`nvblox_torch_pip_installation`. If you intend to interface with ``nvblox`` from Python, this is the recommended method.

2. :ref:`nvblox_torch_source_installation`. Use this method if you intend to interface with ``nvblox`` via the C++ interface or if your platform does not support ``pip``.

3. :ref:`nvblox_native_installation`. Use this method if you want to install ``nvblox`` outside our provided docker environment.


.. _supported_platforms:

.. _supported_platforms_table:

Supported Platforms
-------------------

The following platforms are supported:

+------------------------+-------------+----------------+----------------+-----------------+
|                        | x86 + dGPU  | JetPack 7.0.X  | JetPack 6.X    | JetPack 5.X (*) |
+========================+=============+================+================+=================+
| ``nvblox_torch (pip)`` | ✅          | ❌             | ❌             | ❌              |
+------------------------+-------------+----------------+----------------+-----------------+
| ``nvblox_torch (src)`` | ✅          | ❌             | ✅             | ❌              |
+------------------------+-------------+----------------+----------------+-----------------+
| ``nvblox C++ (src)``   | ✅          | ✅             | ✅             | ✅              |
+------------------------+-------------+----------------+----------------+-----------------+

We support the systems with the following configurations:

- **x86 + discrete GPU**

  - Ubuntu 20.04, 22.04, 24.04
  - CUDA 11.4 (*) - 13.2
  - GPU with compute capability 7.5 or higher. See `here <https://developer.nvidia.com/cuda/gpus>`__ for a list of GPUs and their compute capabilities.

- **Jetson (ARM64)**

  - (ARM64) Jetpack 5, 6, 7

A minimum NVIDIA driver version is imposed by the version of CUDA you have installed.
See the support table `here <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`__
to find the minimum driver version for your platform.

.. note::

    (*): CUDA 11 and Jetpack5 are deprecated and will be removed in an upcoming release.

.. _nvblox_torch_pip_installation:

Install ``nvblox`` via ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:nvblox_torch_pip_install_code_block:

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

   cd $(python3 -c "import site; print(site.getsitepackages()[0])")/nvblox_torch
   pytest -s

You're all set! You can now run the :doc:`torch_examples_reconstruction` example.


.. _nvblox_torch_source_installation:

Install ``nvblox`` from Source (in Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source installation is recommended for developers who need to modify ``nvblox``
or for platforms that are not supported via ``pip``.
We provide a docker image for building and developing inside.


Build the C++ library
^^^^^^^^^^^^^^^^^^^^^

:nvblox_torch_git_clone_code_block:

Then build and run the docker container:

.. code-block:: bash

    cd nvblox
    ./docker/run_docker.sh

To build the library run

.. tabs::
    .. tab:: x86, JetPack 6

        .. code-block:: bash

            mkdir -p /workspaces/nvblox/build
            cd /workspaces/nvblox/build
            cmake ..
            make -j6

    .. tab:: JetPack 7, JetPack 5

           .. code-block:: bash

            mkdir -p /workspaces/nvblox/build
            cd /workspaces/nvblox/build
            cmake .. -DBUILD_PYTORCH_WRAPPER=0
            make -j6

.. note::

    We are using `ccache` to speed up the build process which may sometimes cause issues when the ccache directory is not writable.
    If you see errors like "`/usr/local/bin/c++ is not able to compile a simple test`"" when building, it may help to exit the container and remove the ccache directory:

    .. code-block:: bash

        rm -rf ~/.ccache

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

    ctest --test-dir /workspaces/nvblox/build

.. note::

    Failing tests due to missing or invalid files usually mean the clone was done without git-lfs. Make sure to install git-lfs before cloning the repository.


Install ``nvblox_torch`` python package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

On supported platforms, install the  ``nvblox_torch`` Python library that was built during the previous step:

.. code-block:: bash

    cd /workspaces/nvblox/nvblox_torch
    pip3 install --editable .

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

    pytest -s /workspaces/nvblox/nvblox_torch

You're all set! Feel free to proceed with one of the following examples:

- :doc:`torch_examples_reconstruction` in Python
- :doc:`core_library_run_an_example` from the C++ library.


.. _nvblox_native_installation:

Install ``nvblox`` from Source (Outside Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These instructions describe how to build the ``nvblox`` core C++ library from source,
outside of our development container. They have been tested on Ubuntu 24.04.
All commands below are relative to the repository root.

.. note::

    This recipe disables the ``pytorch`` wrapper and the ``nvblox_renderer``.
    As a result, the Python ``nvblox_torch`` bindings and the GPU renderer
    (used by some examples for visualization) are not available.
    To build with these features enabled, see the system dependencies in
    ``docker/Dockerfile.deps``, or use the :ref:`nvblox_torch_source_installation`
    for a controlled environment.

:nvblox_torch_git_clone_code_block:

Install the build dependencies. A working CUDA Toolkit installation
(see :ref:`supported_platforms_table` for supported versions) is also required.

.. code-block:: bash

    sudo apt-get update && sudo apt-get install -y \
        cmake git git-lfs build-essential python3-dev

From the repository root, configure and build the core library:

.. code-block:: bash

    mkdir build && cd build
    cmake .. -DBUILD_PYTORCH_WRAPPER=0 -DBUILD_RENDERER=0
    make -j6

(Optional) Verify the installation by running the tests:

.. code-block:: bash

    ctest --test-dir .

You're now ready to :doc:`core_library_run_an_example`.


Advanced Build Options
----------------------

This section details build options for advanced ``nvblox`` users.

Modifying maximum feature size
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library supports integrating generic image features into the reconstructed voxel map.
The maximum supported length of image feature vectors is a compile-time constant which defaults to ``128``.
To change the default, call cmake with the following flag:

.. code-block:: bash

   cmake -DNVBLOX_FEATURE_ARRAY_NUM_ELEMENTS=XYZ ..

Note that increasing this number will approximately linearly increase memory usage for applications using deep
feature mapping.

Building for Post-CXX11 ABI
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The library is built with the pre-cxx11 ABI by default in order to maintain compatibility with manylinux201X wheels.
To build with the post cxx11 ABI, call cmake with the following flag:

.. code-block:: bash

   cmake -DPRE_CXX11_ABI_LINKABLE=OFF ..

Disabling ``pytorch`` wrapper
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you don't need the ``pytorch`` wrapper, or you're on a system without ``pytorch`` installed,
you can disable it by calling cmake with the following flag:

.. code-block:: bash

   cmake -DBUILD_PYTORCH_WRAPPER=0 ..

Other ``docker`` containers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We build and test in the following ``docker`` images, so if you would like to install
in a ``docker``, and don't want to use our development ``docker``, these are guaranteed to work.

- ``nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04``
- ``nvcr.io/nvidia/cuda:12.6.1-devel-ubuntu22.04``
- ``nvcr.io/nvidia/cuda:11.8.0-devel-ubuntu22.04``

Build a Redistributable Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

By default, the ``nvblox`` library only builds for the Compute Capability (CC)
of the GPU in the machine it's being built on.
Sometimes it is desirable to build a library that can be used across multiple
machines that contain GPUs with different architectures.
We, for example, build ``nvblox`` for several architectures for packaging
into our ``pip`` package ``nvblox_torch``, such that it can be used on a
variety of machines.

To build binaries that can be used across multiple machines like this, you can
use the ``CMAKE_CUDA_ARCHITECTURE`` flag and set it to a semicolon-separated
list of architectures to support.

For example, to build for Compute Capability (CC) 7.2 and 7.5, you would run:

.. code-block:: bash

    cmake .. -DCMAKE_CUDA_ARCHITECTURES=75;72

Building with Bazel
~~~~~~~~~~~~~~~~~~~

As an alternative to CMake, ``nvblox`` can be built using `Bazel <https://bazel.build/>`_.

.. note::

    Bazel support has the following limitations:

    - **Core C++ library only**: The PyTorch wrapper (``nvblox_torch``) is not supported with Bazel.
    - **Limited platform support**: Tested on Ubuntu 24.04 with GCC 13 x86_64.
    - **Experimental**: Bazel support is newer and less tested than the CMake build system.

To build with Bazel:

.. code-block:: bash

    # Install Bazel (if not already installed)
    # See https://bazel.build/install for installation instructions

    # Build the core library
    bazel build //:nvblox

    # Run tests
    bazel test //nvblox/tests/...

    # Build for aarch64 (experimental)
    bazel build --config arm64 //:nvblox

Build configuration options are defined in ``.bazelrc``. Additional configurations include:

- ``--config asan``: Build with Address Sanitizer
- ``--config tsan``: Build with Thread Sanitizer
- ``--config ubsan``: Build with Undefined Behavior Sanitizer

For more details on the Bazel build system configuration, see the ``.bazelrc`` and ``MODULE.bazel`` files in the repository root.
