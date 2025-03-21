# Library interface

In this page give some brief details of how to interact with nvblox on a library level. For doxygen generated API docs see [our readthedocs page](https://nvblox.readthedocs.io/en/latest/index.html).

## High-level Interface

The top level interface is the [Mapper](@ref nvblox::Mapper) class.

```bash
const float voxel_size_m = 0.05;
const MemoryType memory_type = MemoryType::kDevice;
Mapper(voxel_size_m, memory_type);
```

This creates a mapper, which also allocates an empty map. Here we specify that voxels will be 5cm is size, and will be stored on the GPU (device).

The mapper has methods [integrateDepth](@ref nvblox::Mapper::integrateDepth) and [integrateColor](@ref nvblox::Mapper::integrateColor) for adding depth and color images to the reconstruction:

```bash
mapper.integrateDepth(depth_image, T_L_C, camera);
mapper.integrateColor(color_image, T_L_C, camera);
```

The input [image](@ref nvblox::DepthImage) `depth_image`, the [camera pose](@ref nvblox::Transform) `T_L_C`, and the [camera intrinsic model](@ref nvblox::Camera) `camera` need to be supplied by the user of nvblox.

The function call above integrates the observations into a 3D TSDF voxel grid.
The TSDF is rarely the final desired output and usually we would like to generate a Euclidian Signed Distance Function (ESDF) for pathplanning, or to generate a mesh to view the reconstruction, from the TSDF.
Mapper includes methods for doing this: [updateEsdf](@ref nvblox::Mapper::updateEsdf) and [updateMesh](@ref nvblox::Mapper::updateMesh):

```{.bash}
mapper.updateEsdf();
mapper.updateMesh();
```

The word "update" here indicates that these functions don't generate the mesh or ESDF from scratch, but only update what's needed.

We could then save the mesh to disk as a `.ply` file, by using [outputMeshLayerToPly](@ref nvblox::io::outputMeshLayerToPly):

```bash
io::outputMeshLayerToPly(mapper.mesh_layer(), "/path/to/my/cool/mesh.ply");
```

## Accessing Voxels

If you're using nvblox as a library you likely want to work with voxels directly.

Voxels are stored in the class [Layer](@ref nvblox::Layer). A map is composed of multiple layers, which are co-located voxel grids which stored voxels of different types.
A typical map has for example [TSDF](@ref nvblox::TsdfLayer), [ESDF](@ref nvblox::EsdfLayer) and [Color](@ref nvblox::ColorLayer) layers.

Layer provides voxel accessor methods [getVoxels](@ref nvblox::VoxelBlockLayer::getVoxels) and [getVoxelsGPU](@ref nvblox::VoxelBlockLayer::getVoxelsGPU). These will return the caller with a vector of voxels on either the GPU or CPU.

Calling these functions requires the GPU to run a kernel to retrieve voxels from the voxel grid and copy their values into the output vector.
In the `getVoxels` we additionally copy the voxel back from the GPU to host (CPU) memory.

Getting voxels using the functions above is a multistep process internally.
The function has to:
* Call a kernel which translates query positions to voxel memory locations, 
* Copies voxels into an output vector.
* We optionally have to copy the output vector from device to host memory.
 
Therefore, advanced users who want maximum query speed should access voxels directly inside a GPU kernel.
The next sections discusses this process.

## Accessing Voxels on GPU

If you want to write high performance code which uses voxel values directly, you'll likely want to access voxels in GPU kernels.

We illustrate how this is done by a slightly simplified version of the `getVoxels` function described in the last section.

```cpp
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
  const int num_blocks = num_queries / kNumThreads + 1;

  GPULayerView<TsdfBlock> gpu_layer_view = layer.getGpuLayerView(CudaStreamOwning());

  queryVoxelsKernel<<<num_blocks, kNumThreads>>>(
      num_queries, gpu_layer_view.getHash().impl_, layer.block_size(),
      positions_L.data(), voxels_ptr->data(), success_flags_ptr->data());
  checkCudaErrors(cudaDeviceSynchronize(cuda_stream));
  checkCudaErrors(cudaPeekAtLastError());
}
```

The first critical thing that happens in the code above is that we get a GPU view of the hash table representing the map.

```cpp
GPULayerView<TsdfBlock> gpu_layer_view = layer.getGpuLayerView(CudaStreamOwning())
```
The hash table is used in the kernel to transform 3D query locations into memory locations for voxels.

Inside the kernel we have
```cpp
TsdfVoxel* voxel;
getVoxelAtPosition<TsdfVoxel>(block_hash, query_location, block_size, &voxel);
```
which places a pointer to the voxel in `voxel` and returns true if the voxel has been allocated.

For a small example application which queries voxels on the GPU see `/nvblox/examples/src/esdf_query.cu`.
