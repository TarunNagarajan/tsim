#include "../../examples/common.cuh"
__global__ void cache_hint(const int* input, int* output) {
  int value;
  asm volatile("ld.global.L1::evict_last.u32 %0, [%1];" : "=r"(value) : "l"(input + threadIdx.x) : "memory");
  output[threadIdx.x] = value;
}
int main() {
  int input[32], output[32], *a, *b;
  for (int i = 0; i < 32; ++i) input[i] = i * i;
  CUDA(cudaMalloc(&a, sizeof(input))); CUDA(cudaMalloc(&b, sizeof(output)));
  CUDA(cudaMemcpy(a, input, sizeof(input), cudaMemcpyHostToDevice));
  cache_hint<<<1, 32>>>(a, b);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, b, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(a)); CUDA(cudaFree(b));
  for (int i = 0; i < 32; ++i) if (output[i] != input[i]) return 1;
  std::puts("PASS cache_hint n=32");
}
