#include "../../examples/common.cuh"
#include <vector>

struct Result { unsigned selected, sum, ballot; };
__global__ void collective_wait(Result* output) {
  unsigned lane = threadIdx.x, tid = blockIdx.x * 32 + lane, value = tid;
  if (lane == 0) {
    for (unsigned step = 0; step < 2048; ++step)
      asm volatile("add.u32 %0,%0,3;" : "+r"(value));
  }
  unsigned selected = __shfl_sync(0xffffffffu, value, 0);
  unsigned sum = __reduce_add_sync(0xffffffffu, value);
  unsigned ballot = __ballot_sync(0xffffffffu, value & 1);
  output[tid] = {selected, sum, ballot};
}

int main() {
  Result* device; CUDA(cudaMalloc(&device, 128 * sizeof(Result)));
  collective_wait<<<4, 32>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<Result> got(128);
  CUDA(cudaMemcpy(got.data(), device, 128 * sizeof(Result), cudaMemcpyDeviceToHost)); CUDA(cudaFree(device));
  for (unsigned tid = 0; tid < 128; ++tid) {
    unsigned base = tid / 32 * 32;
    unsigned expected_sum = 0;
    for (unsigned other = 0; other < 32; ++other) expected_sum += base + other + (other == 0 ? 3 * 2048 : 0);
    if (got[tid].selected != base + 3 * 2048 || got[tid].sum != expected_sum || got[tid].ballot != 0xaaaaaaaau) {
      std::fprintf(stderr, "collective_wait mismatch tid=%u selected=%u sum=%u ballot=%x\n",
                   tid, got[tid].selected, got[tid].sum, got[tid].ballot); return 1;
    }
  }
  std::puts("PASS collective_wait threads=128 fields=3");
}
