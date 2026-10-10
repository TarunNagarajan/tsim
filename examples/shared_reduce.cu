#include "common.cuh"
__global__ void shared_reduce(int* total) {
  extern __shared__ int values[];
  int tid = threadIdx.x;
  values[tid] = tid + 1;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride; stride /= 2) {
    if (tid < stride) values[tid] += values[tid + stride];
    __syncthreads();
  }
  if (tid == 0) *total = values[0];
}
int main() {
  int* device; int total = 0;
  CUDA(cudaMalloc(&device, sizeof(int)));
  shared_reduce<<<1, 128, 128 * sizeof(int)>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(&total, device, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  if (total != 128 * 129 / 2) return 1;
  std::printf("PASS shared_reduce total=%d\n", total);
}
