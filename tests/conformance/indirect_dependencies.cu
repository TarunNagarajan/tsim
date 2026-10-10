#include "../../examples/common.cuh"
#include <vector>

__device__ __noinline__ int plus_seven(int x) { return x + 7; }
__device__ __noinline__ int minus_five(int x) { return x - 5; }
__global__ void indirect_dependencies(int* output) {
  int tid = threadIdx.x;
  using Function = int (*)(int);
  Function function = tid & 1 ? plus_seven : minus_five;
  // Preserve an actual function-pointer call through compiler optimization.
  asm volatile("" : "+l"(function));
  output[tid] = function(tid);
}
int main() {
  constexpr int n = 64;
  int* device; CUDA(cudaMalloc(&device, n * sizeof(int)));
  indirect_dependencies<<<1, n>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<int> output(n);
  CUDA(cudaMemcpy(output.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int i = 0; i < n; ++i)
    if (output[i] != (i & 1 ? i + 7 : i - 5)) {
      std::fprintf(stderr, "indirect_dependencies mismatch lane=%d\n", i); return 1;
    }
  std::printf("PASS indirect_dependencies lanes=%d\n", n);
}
