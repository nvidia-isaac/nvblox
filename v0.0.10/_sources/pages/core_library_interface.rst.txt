Core Library Interface
======================

In this page we give some details of how to interact with ``nvblox`` on a library level.

High-level Interface
--------------------

The top-level interface is the ``Mapper`` class.

.. code-block:: cpp

    const float voxel_size_m = 0.05;
    const MemoryType memory_type = MemoryType::kDevice;
    Mapper(voxel_size_s, memory_type);

This creates a mapper that, initially, holds an empty map.
Here we specify that voxels will be 5cm in side length, and will be stored on the GPU (device).

The mapper has methods ``integrateDepth`` and ``integrateColor`` for
adding depth and color images to the reconstruction:

.. code-block:: cpp

    mapper.integrateDepth(depth_image, T_L_C, camera);
    mapper.integrateColor(color_image, T_L_C, camera);

The input ``DepthImage`` ``depth_image``, the ``Transform`` ``T_L_C``, and the ``Camera``
``camera`` need to be supplied by the user of nvblox.

The function calls above integrate the observations into a 3D TSDF voxel grid.
The TSDF is rarely the final desired output and usually we would like to generate a
Euclidean Signed Distance Function (ESDF) for path-planning, or to generate a mesh
to visualize the reconstruction.

The ``Mapper`` class includes methods for doing this: ``updateEsdf`` and ``updateColorMesh``:

.. code-block:: cpp

    mapper.updateEsdf();
    mapper.updateColorMesh();

The word "update" here indicates that these functions don't generate the mesh
or ESDF from scratch, but only update what's needed (typically this is voxels which
were affected by the integration of the new observations since the last update).

We can then save the mesh to disk as a ``.ply`` file, by using ``outputColorMeshLayerToPly``:

.. code-block:: cpp

    io::outputColorMeshLayerToPly(mapper.color_mesh_layer(), "/path/to/my/cool/mesh.ply");

This ``.ply`` can be viewed by many mesh viewers, for example ``meshlab``, ``cloudcompare``,
or ``open3d``.


Accessing Voxels
----------------

We provide two methods for accessing voxels:

1. :ref:`accessing-voxels-via-copying`
2. :ref:`accessing-voxels-without-copying`

Accessing voxels via copying is the easiest way to get voxel values, and can be done
by only writing CPU code only (internally the GPU is called by ``nvblox`` but this is hidden
from the user).
However, because this method requires copying, it doesn't reach peak performance.
Accessing voxels without copying requires writing some GPU code but is *blazingly fast*.


.. _accessing-voxels-via-copying:

Accessing Voxels Via Copying (on CPU and the GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An ``nvblox`` map is composed of multiple ``Layers``, which are co-located voxel grids which
store voxels of different types.
Voxels are stored in the class ``Layer``.
See :doc:`technical_details` for more details.
A typical map has for example ``TsdfLayer``, ``EsdfLayer`` and ``ColorLayer`` layers.

The ``Layer`` class provides voxel accessor methods:

- ``getVoxels()`` and
- ``getVoxelsGPU()``.

These will return the caller with a vector of voxels on either the GPU or CPU.
Under the hood calling these functions will cause the GPU to run a kernel to retrieve voxels
from the voxel grid, and copy their values into the output vector.
In the ``getVoxels()`` function we additionally copy the voxel back from the device (GPU)
to host (CPU) memory.

Below we show an example of retrieving a single voxel at ``[x, y, z] = [0, 0, 0]`` using
this interface:

.. code-block:: cpp

    # The 3D location of the voxel we want to query.
    std::vector<Vector3f> query_positions = {
        Vector3f(0.0f, 0.0f, 0.0f),
    };

    # Setup the output (on the CPU).
    std::vector<TsdfVoxel> voxels;
    std::vector<bool> success_flags;

    # Perform the query + copy.
    layer.getVoxels(query_positions, &voxels, &success_flags);

    # Check that the query was successful.
    if (success_flags[0]) {
        const TsdfVoxel voxel = voxels[0];
        std::cout << "Voxel at (0, 0, 0)"
                  << " has distance: " << voxel.distance
                  << " and weight: " << voxel.weight
                  << std::endl;
    }


.. _accessing-voxels-without-copying:


Accessing Voxels Without Copying (on GPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you want to write high-performance code which uses voxel values directly,
you'll likely want to access voxels in GPU kernels.

We illustrate how this is done here by implementing a toy example.
In particular, we implement a function which copies voxel values into a ``device_vector``.
The full example application which you can build and run can be found at
`esdf_query.cu <https://github.com/nvidia-isaac/nvblox/blob/public/nvblox/examples/src/esdf_query.cu>`_.


The code looks like:

.. code-block:: cpp

    __global__ void queryVoxelsKernel(
        int num_queries, Index3DDeviceHashMapType<TsdfBlock> block_hash,
        float block_size, const Vector3f* query_locations_ptr,
        TsdfVoxel* voxels_ptr, bool* success_flags_ptr) {
      const int idx = threadIdx.x + blockIdx.x * blockDim.x;
      if (idx >= num_queries) {
        return;
      }
      const Vector3f query_location = query_locations_ptr[idx];

      TsdfVoxel* voxel;
      if (!getVoxelAtPosition<TsdfVoxel>(block_hash, query_location, block_size,
                                         &voxel)) {
        success_flags_ptr[idx] = false;
      } else {
        success_flags_ptr[idx] = true;
        voxels_ptr[idx] = *voxel;
      }
    }

    void getVoxelsGPU(
        const TsdfLayer layer,
        const device_vector<Vector3f>& positions_L,
        device_vector<TsdfVoxel>* voxels_ptr,
        device_vector<bool>* success_flags_ptr) const {

      const int num_queries = positions_L.size();

      voxels_ptr->resize(num_queries);
      success_flags_ptr->resize(num_queries);

      constexpr int kNumThreads = 512;
      const int num_blocks = divideRoundUp(num_queries, kNumThreads);

      GPULayerView<TsdfBlock> gpu_layer_view = layer.getGpuLayerView(CudaStreamOwning());

      queryVoxelsKernel<<<num_blocks, kNumThreads>>>(
          num_queries, gpu_layer_view.getHash().impl_, layer.block_size(),
          positions_L.data(), voxels_ptr->data(), success_flags_ptr->data());
      checkCudaErrors(cudaDeviceSynchronize(cuda_stream));
      checkCudaErrors(cudaPeekAtLastError());
    }

The first critical thing that happens in the code above is that we get a GPU view
of the hash table representing the map.

.. code-block:: cpp

    GPULayerView<TsdfBlock> gpu_layer_view = layer.getGpuLayerView(CudaStreamOwning())

The hash table is used in the kernel to transform 3D query locations into
memory locations for voxels.

Inside the kernel we have

.. code-block:: cpp

    TsdfVoxel* voxel;
    getVoxelAtPosition<TsdfVoxel>(block_hash, query_location, block_size, &voxel);

which places a pointer to the voxel in ``voxel`` and returns true if the voxel has been allocated.

If the query was successful, we copy the voxel value into the output vector in global memory.

.. code-block:: cpp

    voxels_ptr[idx] = *voxel;

After the kernel has finished the user is left with a ``device_vector``
populated with voxel values.
