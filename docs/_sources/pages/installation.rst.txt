Installation
============

This page describes how to install :ref:`nvblox_torch_installation` (python) and :ref:`nvblox_installation` (c++).

.. _supported_platforms:

.. _supported_platforms_table:

Supported Platforms
-------------------

The following platforms are supported:

+-------------------+-------------+----------------+
|                   | x86 + dGPU  | Jetson (ARM64) |
+===================+=============+================+
| ``nvblox_torch``  | ✅          | ❌             |
+-------------------+-------------+----------------+
| ``nvblox``        | ✅          | ✅             |
+-------------------+-------------+----------------+


We support the systems with the following configurations:

- **x86 + discrete GPU**

  - Ubuntu 20.04, 22.04, 24.04
  - CUDA 11.4 - 12.8

- **Jetson (ARM64)**

  - (ARM64) Jetpack 5, 6

A minimum NVIDIA driver version is imposed by the version of CUDA you have installed.
See the support table `here <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_
to find the minimum driver version for your platform.


.. _nvblox_torch_installation:

``nvblox_torch``
----------------

There are two ways to install ``nvblox_torch``:

1. :ref:`nvblox_torch_pip_installation`
2. :ref:`nvblox_torch_source_installation`

``pip`` is the preferred way to install ``nvblox_torch`` on :ref:`supported_platforms`.
Source installation is only recommended for developers who need to modify ``nvblox_torch``
or for platforms that are not supported via ``pip``.


.. _nvblox_torch_pip_installation:

Install ``nvblox_torch`` via ``pip``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:nvblox_torch_pip_install_code_block:

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

   cd $(python3 -c "import site; print(site.getsitepackages()[0])")/nvblox_torch
   pytest -s

You're all set! You can now run the :doc:`torch_examples_reconstruction` example.


.. _nvblox_torch_source_installation:

Install ``nvblox_torch`` from Source (in Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The source installation is recommended for developers who need to modify ``nvblox_torch``
or for platforms that are not supported via ``pip``.
We provide a docker image for building and developing inside.

First clone the repository:

:nvblox_torch_git_clone_code_block:

Then build and run the docker container:

.. code-block:: bash

    cd nvblox
    ./docker/run_docker.sh

To build the c++ library run

.. code-block:: bash

    mkdir -p /workspaces/nvblox/build
    cd /workspaces/nvblox/build
    cmake ..
    make -j${nproc}

To install nvblox_torch in development/editable mode run

.. code-block:: bash

    cd /workspaces/nvblox/nvblox_torch
    pip3 install -e .

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

    cd /workspaces/nvblox/nvblox_torch
    pytest -s

You're all set! You can now :doc:`torch_examples_reconstruction`.


.. _nvblox_installation:

``nvblox``
----------

We support two installation methods for building the ``nvblox`` c++ library:

1. :ref:`nvblox_docker_installation` (recommended)
2. :ref:`nvblox_native_installation`

After installing either way, you're ready to :doc:`core_library_run_an_example`.

.. _nvblox_docker_installation:

Install ``nvblox`` from Source (in Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The steps to build ``nvblox`` in the development container are the same as the
instructions in :ref:`nvblox_torch_source_installation`.
``nvblox`` is built inside our development container as the first part of
installing ``nvblox_torch``.

One difference is that on Jetson platforms we need to disable building of the ``pytorch`` wrapper,
which is (currently) only supported on x86 platforms. Also note that the Jetson docker build only
supports Jetpack 6.2. The modified (jetson) and unmodified (x86) building commands are:

.. tabs::

    .. tab:: x86

        .. code-block:: bash

            mkdir -p /workspaces/nvblox/build
            cd /workspaces/nvblox/build
            cmake ..
            make -j${nproc}

    .. tab:: Jetson (ARM64)

        .. code-block:: bash

            mkdir -p /workspaces/nvblox/build
            cd /workspaces/nvblox/build
            cmake .. -DBUILD_PYTORCH_WRAPPER=0
            make -j${nproc}

(Optional) To confirm building was a success, run the tests:

.. code-block:: bash

    cd nvblox
    ctest

You're now ready to  :doc:`core_library_run_an_example`.


.. _nvblox_native_installation:

Install ``nvblox`` from Source (Outside Docker)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These instructions describe how to install the ``nvblox`` core library from source, outside
of our development container.

.. note::

    We recommend using the :ref:`nvblox_docker_installation` as it will handle all the
    dependencies for you.
    The docker image sets up a controlled environment in which we know things work.
    While we've tested the following instructions on many systems
    (see :ref:`supported_platforms_table`), results may vary.

To start, install our dependencies

.. code-block:: bash

    sudo apt-get update && sudo apt-get install cmake git jq gnupg apt-utils software-properties-common build-essential sudo python3-pip wget sudo git python3-dev git-lfs

Note that for Ubuntu 20.04, we need to install a more recent version of ``cmake``
than is available in the default repositories.
We provide a script to add the relevant repositories and install a more recent
version in: ``docker/install_cmake.sh``.
Note that running this script will replace any previously installed version of ``cmake``.

Now follow the instructions in :ref:`nvblox_docker_installation`
to build the code and run the tests.

You're now ready to  :doc:`core_library_run_an_example`.


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
