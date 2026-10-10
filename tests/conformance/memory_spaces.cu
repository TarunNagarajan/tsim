#include "../../examples/common.cuh"
#include <vector>

__constant__ int coefficients[4];
__device__ int launches;
__global__ void spaces(int* output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  volatile int scratch[37];
  for (int i = 0; i < 37; ++i) scratch[i] = (tid + 3) * (i + 1) - coefficients[i % 4];
  int sum = 0;
  for (int i = 0; i < 37; ++i) sum += scratch[(i * 13) % 37];
  output[tid] = sum;
  if (tid == 0) atomicAdd(&launches, 1);
}
int main() {
  constexpr int n = 67;
  int host_coefficients[] = {3, -5, 11, -2}, zero = 0;
  CUDA(cudaMemcpyToSymbol(coefficients, host_coefficients, sizeof(host_coefficients)));
  CUDA(cudaMemcpyToSymbol(launches, &zero, sizeof(zero)));
  int* device; CUDA(cudaMalloc(&device, n * sizeof(int)));
  for (int launch = 0; launch < 3; ++launch) spaces<<<2, 64>>>(device, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<int> got(n); int count = 0;
  CUDA(cudaMemcpy(got.data(), device, n * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpyFromSymbol(&count, launches, sizeof(count))); CUDA(cudaFree(device));
  for (int tid = 0; tid < n; ++tid) {
    int expected = 0;
    for (int i = 0; i < 37; ++i) expected += (tid + 3) * (i + 1) - host_coefficients[i % 4];
    if (got[tid] != expected) { std::fprintf(stderr, "memory_spaces mismatch tid=%d got=%d expected=%d\n", tid, got[tid], expected); return 1; }
  }
  if (count != 3) return 1;
  std::puts("PASS memory_spaces n=67 local=37 launches=3");
}
