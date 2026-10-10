#include "common.cuh"
__global__ void divergent_paths(int* output) {
  int i = threadIdx.x;
  // Volatile work prevents the compiler replacing the branches with a select.
  volatile int value = i;
  if (i & 1) {
    for (int j = 0; j < 3; ++j) value = value * 2 + 1;
  } else {
    for (int j = 0; j < 2; ++j) value = value + 7;
  }
  output[i] = value;
}
int main() {
  int* device; int output[64];
  CUDA(cudaMalloc(&device, sizeof(output)));
  divergent_paths<<<1, 64>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) if (output[i] != ((i & 1) ? 8 * i + 7 : i + 14)) return 1;
  CUDA(cudaFree(device));
  std::printf("PASS divergence even=%d odd=%d\n", output[0], output[1]);
}
