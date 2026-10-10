#include "../../examples/common.cuh"
#include <vector>
__global__ void collective_nonvirtual_wait(unsigned* output, const volatile unsigned* masks) {
  unsigned tid = blockIdx.x * 32 + threadIdx.x, value = tid, selected;
  if (!threadIdx.x) for (unsigned step = 0; step < 2048; ++step)
    asm volatile("add.u32 %0,%0,3;" : "+r"(value));
  unsigned members = masks[tid + (value >> 16)];
  asm volatile("shfl.sync.idx.b32 %0,17,0,31,%1;" : "=r"(selected) : "r"(members));
  output[tid] = selected + value;
}
int main() {
  unsigned *device, *device_masks;
  CUDA(cudaMalloc(&device, 128 * sizeof(unsigned))); CUDA(cudaMalloc(&device_masks, 128 * sizeof(unsigned)));
  std::vector<unsigned> masks(128, 0xffffffffu);
  CUDA(cudaMemcpy(device_masks, masks.data(), masks.size() * sizeof(unsigned), cudaMemcpyHostToDevice));
  collective_nonvirtual_wait<<<4, 32>>>(device, device_masks); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<unsigned> got(128); CUDA(cudaMemcpy(got.data(), device, got.size() * sizeof(unsigned), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device)); CUDA(cudaFree(device_masks));
  for (unsigned tid = 0; tid < got.size(); ++tid) if (got[tid] != 17 + tid + (tid % 32 == 0 ? 3 * 2048 : 0)) {
    std::fprintf(stderr, "collective_nonvirtual_wait mismatch tid=%u got=%u\n", tid, got[tid]); return 1;
  }
  std::puts("PASS collective_nonvirtual_wait threads=128 immediate_data=17");
}
