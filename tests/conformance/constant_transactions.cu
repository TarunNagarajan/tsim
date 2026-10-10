#include "../../examples/common.cuh"
__constant__ int table[32];
__global__ void constant_words(int* output) {
  unsigned tid = threadIdx.x;
  output[tid] = 2 * table[tid] - 3;
}
int main() {
  int input[32], output[32], *device;
  for (int i = 0; i < 32; ++i) input[i] = i * i + 7;
  CUDA(cudaMemcpyToSymbol(table, input, sizeof(input)));
  CUDA(cudaMalloc(&device, sizeof(output)));
  constant_words<<<1, 32>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != 2 * input[i] - 3) return 1;
  std::puts("PASS constant_transactions distinct=32");
}
