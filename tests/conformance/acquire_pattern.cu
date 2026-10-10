#include "../../examples/common.cuh"
#include <cuda/atomic>
__global__ void pattern(int* output) {
  __shared__ int payload[32], flag, read_first, replaced;
  int tid = threadIdx.x, lane = tid % 32;
  if (tid == 0) flag = read_first = replaced = 0;
  __syncthreads();
  cuda::atomic_ref<int, cuda::thread_scope_block> ready(flag), seen(read_first), done(replaced);
  if (tid < 32) {
    payload[lane] = lane * 19 + 5;
    __syncwarp();
    if (lane == 0) ready.store(1, cuda::memory_order_release);
  } else if (tid < 64) {
    while (!seen.load(cuda::memory_order_relaxed)) {}
    if (lane == 0) {
      ready.store(2, cuda::memory_order_relaxed);
      done.store(1, cuda::memory_order_relaxed);
    }
  } else {
    while (!ready.load(cuda::memory_order_relaxed)) {}
    __syncwarp(); // every consumer has observed the original publication
    if (lane == 0) seen.store(1, cuda::memory_order_relaxed);
    while (!done.load(cuda::memory_order_relaxed)) {}
    while (ready.load(cuda::memory_order_acquire) < 2) {}
    output[lane] = payload[lane];
  }
}
int main() {
  int* device; int output[32]; CUDA(cudaMalloc(&device, sizeof(output)));
  pattern<<<1, 96>>>(device); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != i * 19 + 5) return 1;
  std::puts("PASS acquire_pattern consumers=32");
}
