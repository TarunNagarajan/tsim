#include "../../examples/common.cuh"
#include <cuda/atomic>
__global__ void publication(int* output) {
  __shared__ int payload[32], first, second;
  int tid = threadIdx.x, lane = tid % 32;
  if (tid == 0) first = second = 0;
  __syncthreads();
  cuda::atomic_ref<int, cuda::thread_scope_block> a(first), b(second);
  if (tid < 32) {
    payload[lane] = lane * 13 + 7;
    __syncwarp();
    if (lane == 0) a.store(1, cuda::memory_order_release);
  } else if (tid < 64) {
    while (!a.load(cuda::memory_order_acquire)) {}
    if (lane == 0) b.store(1, cuda::memory_order_release);
  } else {
    while (!b.load(cuda::memory_order_acquire)) {}
    output[lane] = payload[lane];
  }
}
int main() {
  int* device; int output[32]; CUDA(cudaMalloc(&device, sizeof(output)));
  publication<<<1, 96>>>(device); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != i * 13 + 7) return 1;
  std::puts("PASS transitive_publication consumers=32");
}
