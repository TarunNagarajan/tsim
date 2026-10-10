#include "../../examples/common.cuh"
#include <cuda/atomic>
__global__ void rmw_publish(int* output) {
  __shared__ int payload[32], flag, relayed;
  int tid = threadIdx.x, lane = tid % 32;
  if (tid == 0) flag = relayed = 0;
  __syncthreads();
  cuda::atomic_ref<int, cuda::thread_scope_block> ready(flag);
  cuda::atomic_ref<int, cuda::thread_scope_block> done(relayed);
  if (tid < 32) {
    payload[lane] = lane * 17 + 11;
    __syncwarp();
    if (lane == 0) ready.store(1, cuda::memory_order_release);
  } else if (tid < 64) {
    while (ready.load(cuda::memory_order_relaxed) < 1) {}
    if (lane == 0) {
#ifdef RELAY_STORE
      ready.store(2, cuda::memory_order_relaxed);
#else
      ready.fetch_add(1, cuda::memory_order_relaxed);
#endif
      done.store(1, cuda::memory_order_relaxed);
    }
  } else {
    while (!done.load(cuda::memory_order_relaxed)) {}
    while (ready.load(cuda::memory_order_acquire) < 2) {}
    output[lane] = payload[lane];
  }
}
int main() {
  int* device; int output[32]; CUDA(cudaMalloc(&device, sizeof(output)));
  rmw_publish<<<1, 96>>>(device); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != i * 17 + 11) return 1;
  std::puts("PASS rmw_publication consumers=32");
}
