#include "../../examples/common.cuh"
#include <vector>

struct Result { unsigned values[8]; };
__global__ void dynamic_members(Result* output, const volatile unsigned* masks,
                                const volatile unsigned* bits, const volatile int* offsets) {
  unsigned tid = threadIdx.x, lane = tid & 31;
  unsigned members = masks[tid], bitset = bits[tid];
  int offset = offsets[tid];
  Result result{};
  result.values[0] = __ballot_sync(members, lane % 3 == 0);
  result.values[1] = __all_sync(members, lane < 32);
  result.values[2] = __any_sync(members, lane % 16 == 7);
  result.values[3] = __match_any_sync(members, lane % 4);
  result.values[4] = __reduce_add_sync(members, lane + 1);
  result.values[5] = __shfl_xor_sync(members, tid + 10, 1, 16);
  asm volatile("fns.b32 %0,%1,%2,%3;" : "=r"(result.values[6]) : "r"(bitset), "r"(lane), "r"(offset));
  asm volatile("fns.b32 %0,%1,%2,%3;" : "=r"(result.values[7]) : "r"(bitset), "r"(lane), "r"(-offset));
  output[tid] = result;
}

unsigned reference_bit(unsigned bits, unsigned base, int offset) {
  if (!offset) return bits & (1u << base) ? base : 0xffffffffu;
  std::vector<unsigned> candidates;
  for (unsigned bit = 0; bit < 32; ++bit)
    if ((bits & (1u << bit)) && (offset > 0 ? bit >= base : bit <= base)) candidates.push_back(bit);
  unsigned rank = offset > 0 ? unsigned(offset) : unsigned(-offset);
  if (rank > candidates.size()) return 0xffffffffu;
  return offset > 0 ? candidates[rank - 1] : candidates[candidates.size() - rank];
}

int main() {
  std::vector<unsigned> masks(64), bits(64); std::vector<int> offsets(64);
  for (unsigned tid = 0; tid < 64; ++tid) {
    masks[tid] = 0xffffu << ((tid % 32) / 16 * 16);
    bits[tid] = 0x55555555u ^ (1u << (tid % 32)); offsets[tid] = int(tid % 5);
  }
  Result* device; unsigned *device_masks, *device_bits; int* device_offsets;
  CUDA(cudaMalloc(&device, 64 * sizeof(Result))); CUDA(cudaMalloc(&device_masks, 64 * sizeof(unsigned)));
  CUDA(cudaMalloc(&device_bits, 64 * sizeof(unsigned))); CUDA(cudaMalloc(&device_offsets, 64 * sizeof(int)));
  CUDA(cudaMemcpy(device_masks, masks.data(), 64 * sizeof(unsigned), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(device_bits, bits.data(), 64 * sizeof(unsigned), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(device_offsets, offsets.data(), 64 * sizeof(int), cudaMemcpyHostToDevice));
  dynamic_members<<<1, 64>>>(device, device_masks, device_bits, device_offsets);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<Result> got(64);
  CUDA(cudaMemcpy(got.data(), device, 64 * sizeof(Result), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device)); CUDA(cudaFree(device_masks)); CUDA(cudaFree(device_bits)); CUDA(cudaFree(device_offsets));
  for (unsigned tid = 0; tid < 64; ++tid) {
    unsigned lane = tid % 32, ballot = 0, match = 0, sum = 0;
    for (unsigned other = 0; other < 32; ++other) if (masks[tid] & (1u << other)) {
      if (other % 3 == 0) ballot |= 1u << other;
      if (other % 4 == lane % 4) match |= 1u << other;
      sum += other + 1;
    }
    Result expected{{ballot, 1, 1, match, sum, (tid ^ 1) + 10,
      reference_bit(bits[tid], lane, offsets[tid]), reference_bit(bits[tid], lane, -offsets[tid])}};
    for (unsigned field = 0; field < 8; ++field) if (got[tid].values[field] != expected.values[field]) {
      std::fprintf(stderr, "dynamic_members mismatch tid=%u field=%u got=%u expected=%u\n",
                   tid, field, got[tid].values[field], expected.values[field]); return 1;
    }
  }
  std::puts("PASS dynamic_members threads=64 fields=8");
}
