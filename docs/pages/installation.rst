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

+------------------------+-------------+----------------+-------------+
|                        | x86 + dGPU  | JetPack 6.X    | JetPack 5.X |
+========================+=============+================+=============+
| ``nvblox_torch (pip)`` | ✅          | ❌             | ❌          |
+------------------------+-------------+----------------+-------------+
| ``nvblox_torch (src)`` | ✅          | ✅             | ❌          |
+------------------------+-------------+----------------+-------------+
| ``nvblox C++ (src)``   | ✅          | ✅             | ✅          |
+------------------------+-------------+----------------+-------------+

We support the systems with the following configurations:

- **x86 + discrete GPU**

  - Ubuntu 20.04, 22.04, 24.04
  - CUDA 11.4 - 12.8

- **Jetson (ARM64)**

  - (ARM64) Jetpack 5, 6

A minimum NVIDIA driver version is imposed by the version of CUDA you have installed.
See the support table `here <https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_
to find the minimum driver version for your platform.


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

First clone the repository:

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
            make -j${nproc}

    .. tab:: JetPack 5

           .. code-block:: bash

            mkdir -p /workspaces/nvblox/build
            cd /workspaces/nvblox/build
            cmake .. -DBUILD_PYTORCH_WRAPPER=0
            make -j${nproc}

(Optional) You can verify the installation by running our tests:

.. code-block:: bash

    ctest --test-dir /workspaces/nvblox/build/nvblox

.. note::

    We are using `ccache` to speed up the build process which may sometimes cause issues when the ccache directory is not writable.
    If you see errors like "`/usr/local/bin/c++ is not able to compile a simple test`"" when building, make sure that the mounted `~/.ccache` is writable for the current user:

    .. code-block:: bash

        sudo chmod +rw ~/.ccache

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

These instructions describe how to install the ``nvblox`` core library from source, outside
of our development container.

.. note::

    We recommend using the :ref:`nvblox_torch_source_installation` as it will handle all the
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

Now follow the instructions in :ref:`nvblox_torch_source_installation`
to build the code and run the tests.

If you are using a Jetson and want to use the ``pytorch`` wrapper, you will need to install the CUDA-enabled versions of ``torch`` and ``torchvision``. See `this page <https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html>`_ for more details.

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
