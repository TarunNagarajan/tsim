#include "../../examples/common.cuh"
__global__ void shared_prefix(int* output) {
  __shared__ volatile int tag;
  extern __shared__ __align__(16) int values[];
  unsigned tid = threadIdx.x;
  if (tid == 0) tag = 9;
  if (tid < 31) values[tid] = 2 * tid + 1;
  __syncthreads();
  output[tid] = tag + (tid < 31 ? values[tid] : 0);
}
int main() {
  int output[32], *device;
  CUDA(cudaMalloc(&device, sizeof(output)));
  shared_prefix<<<1, 32, 124>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != 9 + (i < 31 ? 2 * i + 1 : 0)) return 1;
  std::puts("PASS shared_prefix alignment=16");
}
