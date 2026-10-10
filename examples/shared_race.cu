#include "common.cuh"
__global__ void shared_race(int* output) {
  __shared__ int values[64];
  int tid = threadIdx.x;
  values[tid] = tid + 1;
  // Deliberately missing __syncthreads(): warp 0 reads warp 1's writes.
  if (tid < 32) output[tid] = values[tid + 32];
}
int main() {
  int* device; CUDA(cudaMalloc(&device, 32 * sizeof(int)));
  shared_race<<<1, 64>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaFree(device));
}
