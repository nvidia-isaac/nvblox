#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000000
#define MAX_ERR 1e-6

__global__ void vector_add(float* out, float* a, float* b, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  // Handling arbitrary vector size
  if (tid < n) {
    out[tid] = a[tid] + b[tid];
  }
}

int main() {
  float *a, *b, *out;
  float *d_a, *d_b, *d_out;

  // Allocate host memory
  a = (float*)malloc(sizeof(float) * N);
  b = (float*)malloc(sizeof(float) * N);
  out = (float*)malloc(sizeof(float) * N);

  // Initialize host arrays
  for (int i = 0; i < N; i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // Allocate device memory
  cudaMalloc((void**)&d_a, sizeof(float) * N);
  cudaMalloc((void**)&d_b, sizeof(float) * N);
  cudaMalloc((void**)&d_out, sizeof(float) * N);

  // Transfer data from host to device memory
  cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

  // Executing kernel
  int block_size = 256;
  int grid_size = ((N + block_size) / block_size);
  vector_add<<<grid_size, block_size>>>(d_out, d_a, d_b, N);

  // Transfer data back to host memory
  cudaMemcpy(out, d_out, sizeof(float) * N, cudaMemcpyDeviceToHost);

  // Verification
  for (int i = 0; i < N; i++) {
    assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
  }
  printf("out[0] = %f\n", out[0]);
  printf("PASSED\n");

  // Deallocate device memory
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_out);

  // Deallocate host memory
  free(a);
  free(b);
  free(out);
}