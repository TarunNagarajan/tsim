#include "../../examples/common.cuh"
#include <vector>

__device__ __noinline__ int transform(int x, int lane) {
  if (lane % 3 == 0) return x * 3 - 7;
  if (lane % 3 == 1) return x + 19;
  return x * x - 11;
}
__global__ void calls(int* output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  int value = tid % 13 - 6;
  for (int i = 0; i < tid % 7 + 1; ++i) {
    value = transform(value % 23, tid + i);
    if ((tid + i) % 5 == 0) value -= 3;
  }
  output[tid] = value;
}
int main() {
  constexpr int n = 197;
  int* device; CUDA(cudaMalloc(&device, n * sizeof(int)));
  calls<<<4, 64>>>(device, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<int> got(n);
  CUDA(cudaMemcpy(got.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  long long sum = 0;
  for (int tid = 0; tid < n; ++tid) {
    int expected = tid % 13 - 6;
    for (int i = 0; i < tid % 7 + 1; ++i) {
      int x = expected % 23, lane = tid + i;
      expected = lane % 3 == 0 ? x * 3 - 7 : lane % 3 == 1 ? x + 19 : x * x - 11;
      if (lane % 5 == 0) expected -= 3;
    }
    if (got[tid] != expected) { std::fprintf(stderr, "device_calls mismatch tid=%d got=%d expected=%d\n", tid, got[tid], expected); return 1; }
    sum += got[tid];
  }
  std::printf("PASS device_calls n=%d sum=%lld\n", n, sum);
}
