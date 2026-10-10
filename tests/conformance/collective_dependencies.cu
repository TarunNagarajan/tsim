#include "../../examples/common.cuh"
#include <vector>

__global__ void collective_dependencies(unsigned* output) {
  unsigned tid = blockIdx.x * blockDim.x + threadIdx.x, value;
  asm volatile("{ .reg .pred p; .reg .b32 v, parity;\n"
               "and.b32 parity,%1,1;\n"
               "setp.eq.u32 p,parity,0;\n"
               "@p add.u32 v,%1,100;\n"
               "@!p add.u32 v,%1,200;\n"
               "shfl.sync.bfly.b32 v,v,1,31,-1;\n"
               "mov.b32 %0,v; }" : "=r"(value) : "r"(tid));
  output[tid] = value;
}

int main() {
  unsigned* device; CUDA(cudaMalloc(&device, 128 * sizeof(unsigned)));
  collective_dependencies<<<2, 64>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<unsigned> got(128);
  CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (unsigned tid = 0; tid < got.size(); ++tid) {
    unsigned source = tid ^ 1, expected = source + (source & 1 ? 200 : 100);
    if (got[tid] != expected) {
      std::fprintf(stderr, "collective_dependencies mismatch tid=%u got=%u expected=%u\n", tid, got[tid], expected);
      return 1;
    }
  }
  std::puts("PASS collective_dependencies threads=128 in_place=1 distinct_lane_writers=2");
}
