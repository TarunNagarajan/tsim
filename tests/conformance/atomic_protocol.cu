#include "../../examples/common.cuh"
#include <cuda/atomic>
#include <algorithm>
#include <vector>
#ifndef PUBLISH_ORDER
#define PUBLISH_ORDER cuda::memory_order_release
#endif
#ifndef OBSERVE_ORDER
#define OBSERVE_ORDER cuda::memory_order_acquire
#endif

__global__ void protocol(int* output, unsigned* tickets, unsigned* counter) {
  __shared__ int payload[32];
  __shared__ int published;
  int tid = threadIdx.x;
  if (tid == 0) published = 0;
  __syncthreads();
  cuda::atomic_ref<int, cuda::thread_scope_block> ready(published);
  if (tid < 32) {
    payload[tid] = tid * 7 + 3;
    __syncwarp(0xffffffffu);
    if (tid == 0) ready.store(1, PUBLISH_ORDER);
    output[tid] = -1;
  } else {
    while (!ready.load(OBSERVE_ORDER)) {}
    output[tid] = payload[tid - 32];
  }
  tickets[tid] = atomicAdd(counter, 1u);
}
int main() {
  int* device_output; unsigned* device_tickets; unsigned* device_counter;
  CUDA(cudaMalloc(&device_output, 64 * sizeof(int))); CUDA(cudaMalloc(&device_tickets, 64 * sizeof(unsigned)));
  CUDA(cudaMalloc(&device_counter, sizeof(unsigned))); CUDA(cudaMemset(device_counter, 0, sizeof(unsigned)));
  protocol<<<1, 64>>>(device_output, device_tickets, device_counter);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<int> output(64); std::vector<unsigned> tickets(64); unsigned counter;
  CUDA(cudaMemcpy(output.data(), device_output, 64 * sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(tickets.data(), device_tickets, 64 * sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(&counter, device_counter, sizeof(counter), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device_output)); CUDA(cudaFree(device_tickets)); CUDA(cudaFree(device_counter));
  for (int tid = 0; tid < 64; ++tid) if (output[tid] != (tid < 32 ? -1 : (tid - 32) * 7 + 3)) { std::fprintf(stderr, "atomic_protocol payload mismatch tid=%d\n", tid); return 1; }
  std::sort(tickets.begin(), tickets.end());
  for (unsigned i = 0; i < 64; ++i) if (tickets[i] != i) return 1;
  if (counter != 64) return 1;
  std::puts("PASS atomic_protocol consumers=32 tickets=64");
}
