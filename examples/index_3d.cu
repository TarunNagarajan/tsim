#include "common.cuh"
__global__ void index_3d(int* output) {
  unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y = blockIdx.y * blockDim.y + threadIdx.y;
  unsigned z = blockIdx.z * blockDim.z + threadIdx.z;
  output[x + 6 * (y + 4 * z)] = int(x + 100 * y + 10000 * z);
}
int main() {
  int* device; int output[6 * 4 * 4];
  CUDA(cudaMalloc(&device, sizeof(output)));
  index_3d<<<dim3(2, 2, 2), dim3(3, 2, 2)>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int z = 0; z < 4; ++z) for (int y = 0; y < 4; ++y) for (int x = 0; x < 6; ++x)
    if (output[x + 6 * (y + 4 * z)] != x + 100 * y + 10000 * z) return 1;
  std::printf("PASS index_3d elements=%d\n", 6 * 4 * 4);
}
