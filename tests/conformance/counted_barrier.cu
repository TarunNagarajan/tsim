#include "../../examples/common.cuh"
#include <cuda/atomic>
__global__ void counted(int* output) {
  __shared__ int payload, published;
  int tid = threadIdx.x;
  if (tid == 0) published = 0;
  __syncthreads();
  cuda::atomic_ref<int, cuda::thread_scope_block> flag(published);
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(&payload));
  if (tid < 32) {
#ifndef POST_ARRIVAL
    if (tid == 0) asm volatile("st.shared.u32 [%0], %1;" :: "r"(address), "r"(731) : "memory");
    __syncwarp();
#endif
    asm volatile("bar.arrive 1, 64;" ::: "memory");
#ifdef POST_ARRIVAL
    if (tid == 0) asm volatile("st.shared.u32 [%0], %1;" :: "r"(address), "r"(731) : "memory");
#endif
    if (tid == 0) flag.store(1, cuda::memory_order_relaxed);
  } else {
    asm volatile("bar.sync 1, 64;" ::: "memory");
    while (!flag.load(cuda::memory_order_relaxed)) {}
    int observed;
    asm volatile("ld.shared.u32 %0, [%1];" : "=r"(observed) : "r"(address) : "memory");
    output[tid - 32] = observed;
  }
}
int main() {
  int* device; int output[32]; CUDA(cudaMalloc(&device, sizeof(output)));
  counted<<<1, 64>>>(device); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, sizeof(output), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  for (int i = 0; i < 32; ++i) if (output[i] != 731) return 1;
  std::puts("PASS counted_barrier consumers=32");
}
