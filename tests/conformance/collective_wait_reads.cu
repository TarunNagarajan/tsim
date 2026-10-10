#include "../../examples/common.cuh"
#include <vector>

struct Result { unsigned ballot, value; };
template<bool Constant> __global__ void collective_wait_reads(Result* output, const volatile unsigned* masks) {
  unsigned lane = threadIdx.x, tid = blockIdx.x * 32 + lane, value = tid;
  if (!lane) for (unsigned step = 0; step < 2048; ++step)
    asm volatile("add.u32 %0,%0,3;" : "+r"(value));
  // The address depends on the completed loop, preserving a genuine delayed
  // runtime membership load even for a constant vote predicate.
  unsigned members = masks[tid + (value >> 16)];
  unsigned ballot = __ballot_sync(members, Constant ? true : bool(value & 1));
  output[tid] = {ballot, value};
}
int main() {
  Result* device; unsigned* device_masks;
  CUDA(cudaMalloc(&device, 128 * sizeof(Result))); CUDA(cudaMalloc(&device_masks, 128 * sizeof(unsigned)));
  std::vector<unsigned> masks(128, 0xffffffffu);
  CUDA(cudaMemcpy(device_masks, masks.data(), masks.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  for (unsigned constant = 0; constant < 2; ++constant) {
    if (constant) collective_wait_reads<true><<<4, 32>>>(device, device_masks);
    else collective_wait_reads<false><<<4, 32>>>(device, device_masks);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    std::vector<Result> got(128); CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(Result), cudaMemcpyDeviceToHost));
    for (unsigned tid = 0; tid < got.size(); ++tid) {
      unsigned expected = tid + (tid % 32 == 0 ? 3 * 2048 : 0);
      if (got[tid].value != expected || got[tid].ballot != (constant ? 0xffffffffu : 0xaaaaaaaau)) {
        std::fprintf(stderr, "collective_wait_reads mismatch constant=%u tid=%u\n", constant, tid); return 1;
      }
    }
  }
  CUDA(cudaFree(device)); CUDA(cudaFree(device_masks));
  std::puts("PASS collective_wait_reads threads=128 variants=2");
}
