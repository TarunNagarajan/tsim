#include "common.cuh"
__global__ void out_of_bounds(int* pointer) { pointer[threadIdx.x] = int(threadIdx.x); }
int main() {
  int* device; CUDA(cudaMalloc(&device, 4 * sizeof(int)));
  out_of_bounds<<<1, 32>>>(device); // Deliberate fault: 32 threads, only 4 elements.
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaFree(device));
  return 0;
}
