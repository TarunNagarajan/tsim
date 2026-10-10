#include "../../examples/common.cuh"
#include <cstdint>
#include <vector>

__global__ void carry_dependencies(uint32_t* output, int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n) return;
  uint32_t a = 0xfffffff0u + uint32_t(tid), b = 0x30u + uint32_t(tid);
  uint32_t al, ah, dl, dh, ml, mh, filtered, final;
  asm volatile(
    "{\n.reg .u32 al,ah,dl,dh,ml,mh,filtered,final,trash;\n.reg .pred p;\n"
    "add.cc.u32 al, %8, %9;\naddc.u32 ah, 0, 0;\n"
    "sub.cc.u32 dl, %8, %9;\nsubc.u32 dh, 0, 0;\n"
    "mad.lo.cc.u32 ml, %8, %9, 0xffffffff;\nmadc.hi.u32 mh, %8, %9, 0;\n"
    "add.cc.u32 trash, 0xffffffff, 1;\n"
    "setp.eq.u32 p, %10, 0;\nmov.u32 filtered, 20;\n"
    "@p add.cc.u32 trash, 0, 0;\n"
    "@!p addc.u32 filtered, 10, 0;\naddc.u32 final, 30, 0;\n"
    "mov.u32 %0,al;\nmov.u32 %1,ah;\nmov.u32 %2,dl;\nmov.u32 %3,dh;\n"
    "mov.u32 %4,ml;\nmov.u32 %5,mh;\nmov.u32 %6,filtered;\nmov.u32 %7,final;\n}\n"
    : "=r"(al), "=r"(ah), "=r"(dl), "=r"(dh), "=r"(ml), "=r"(mh), "=r"(filtered), "=r"(final)
    : "r"(a), "r"(b), "r"(uint32_t(tid & 1)));
  uint32_t values[8] = {al, ah, dl, dh, ml, mh, filtered, final};
  for (int i = 0; i < 8; ++i) output[tid * 8 + i] = values[i];
}
int main() {
  constexpr int n = 67;
  uint32_t* device; CUDA(cudaMalloc(&device, n * 8 * sizeof(uint32_t)));
  carry_dependencies<<<2, 64>>>(device, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<uint32_t> output(n * 8);
  CUDA(cudaMemcpy(output.data(), device, output.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  for (int tid = 0; tid < n; ++tid) {
    uint32_t a = 0xfffffff0u + uint32_t(tid), b = 0x30u + uint32_t(tid);
    uint64_t sum = uint64_t(a) + b, difference = uint64_t(a) - uint64_t(b);
    uint64_t product = uint64_t(a) * b + 0xffffffffu;
    uint32_t expected[8] = {uint32_t(sum), uint32_t(sum >> 32), uint32_t(difference), uint32_t(difference >> 32),
                            uint32_t(product), uint32_t(product >> 32), uint32_t(tid & 1 ? 11 : 20), uint32_t(tid & 1 ? 31 : 30)};
    for (int i = 0; i < 8; ++i)
      if (output[tid * 8 + i] != expected[i]) {
        std::fprintf(stderr, "carry_dependencies mismatch tid=%d field=%d got=%u expected=%u\n", tid, i, output[tid * 8 + i], expected[i]);
        return 1;
      }
  }
  std::printf("PASS carry_dependencies lanes=%d fields=8\n", n);
}
