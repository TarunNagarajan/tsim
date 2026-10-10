#include "../../examples/common.cuh"
#include <vector>
struct Result { unsigned predicate, mask; };
__global__ void collective_discarded(Result* output) {
  unsigned tid = threadIdx.x, value = tid < 32 ? 7 : tid, predicate, mask;
  asm volatile("{ .reg .pred p; match.all.sync.b32 _|p,%1,-1; selp.u32 %0,1,0,p; }"
               : "=r"(predicate) : "r"(value));
  asm volatile("match.all.sync.b32 %0|_,%1,-1;" : "=r"(mask) : "r"(value));
  output[tid] = {predicate, mask};
}
int main() {
  Result* device; CUDA(cudaMalloc(&device, 64 * sizeof(Result)));
  collective_discarded<<<1, 64>>>(device); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<Result> got(64); CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(Result), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (unsigned tid = 0; tid < got.size(); ++tid) if (got[tid].predicate != (tid < 32 ? 1u : 0u) ||
      got[tid].mask != (tid < 32 ? 0xffffffffu : 0u)) {
    std::fprintf(stderr, "collective_discarded mismatch tid=%u\n", tid); return 1;
  }
  std::puts("PASS collective_discarded threads=64 fields=2 data_and_predicate_sinks=1");
}
