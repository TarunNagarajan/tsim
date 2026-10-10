#include "common.cuh"
__global__ void increment(int* value) { if (threadIdx.x == 0) atomicAdd(value, 1); }
int main() {
  int* device; int result;
  CUDA(cudaMalloc(&device, sizeof(int))); CUDA(cudaMemset(device, 0, sizeof(int)));
  increment<<<1, 32>>>(device);
  increment<<<1, 32>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(&result, device, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  if (result != 2) return 1;
  std::printf("PASS multiple_launches total=%d\n", result);
}
