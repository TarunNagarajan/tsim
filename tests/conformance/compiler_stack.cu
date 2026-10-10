#include "../../examples/common.cuh"
#include <vector>

__device__ __noinline__ int stacked_identity(int x) {
  volatile int values[37];
  for (int i = 0; i < 37; ++i) values[i] = x + i;
  int index = x & 31;
  return values[index] - index;
}
__global__ void nested_stack(int* output) {
  int tid = threadIdx.x;
  volatile int values[17];
  for (int i = 0; i < 17; ++i) values[i] = tid + i;
  int restored = stacked_identity(tid);
  int index = tid & 15;
  output[tid] = values[index] - index + restored;
}
int main() {
  constexpr int n = 64;
  int* device; CUDA(cudaMalloc(&device, n * sizeof(int)));
  nested_stack<<<1, n>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<int> output(n);
  CUDA(cudaMemcpy(output.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int i = 0; i < n; ++i)
    if (output[i] != 2 * i) { std::fprintf(stderr, "compiler_stack mismatch lane=%d\n", i); return 1; }
  std::printf("PASS compiler_stack lanes=%d\n", n);
}
