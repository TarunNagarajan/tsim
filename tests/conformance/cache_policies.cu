#include "../../examples/common.cuh"
__global__ void cache_policies(const int* input, int* output) {
  unsigned tid = threadIdx.x;
  const int* address = input + tid;
  int a, b, c, d;
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(a) : "l"(address) : "memory");
  asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(b) : "l"(address) : "memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(c) : "l"(address) : "memory");
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(d) : "l"(address) : "memory");
  int value = a + b + c + d;
  asm volatile("st.global.cg.u32 [%0], %1;" :: "l"(output + tid), "r"(value) : "memory");
}
int main() {
  int input[32], output[32], *a, *b;
  for (int i = 0; i < 32; ++i) input[i] = 5 * i + 3;
  CUDA(cudaMalloc(&a, sizeof(input))); CUDA(cudaMalloc(&b, sizeof(output)));
  CUDA(cudaMemcpy(a, input, sizeof(input), cudaMemcpyHostToDevice));
  cache_policies<<<1, 32>>>(a, b);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, b, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(a)); CUDA(cudaFree(b));
  for (int i = 0; i < 32; ++i) if (output[i] != 4 * input[i]) return 1;
  std::puts("PASS cache_policies n=32");
}
