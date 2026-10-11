#include "../../examples/common.cuh"
#include <initializer_list>

// Each round changes the operation on a named barrier after completing the
// previous phase. The predicate differs by block, lane, warp and round.
__global__ void reductions(int* output) {
  const unsigned tid = threadIdx.x;
  const unsigned block = blockIdx.x;
  for (unsigned round = 0; round < 7; ++round) {
    const int vote = ((tid + block + round) % 5) < 2;
    unsigned count;
    int every, any;
    asm volatile("{ .reg .pred p, q; setp.ne.s32 p, %1, 0; "
                 "bar.red.popc.u32 %0, 11, p; }"
                 : "=r"(count) : "r"(vote) : "memory");
    asm volatile("bar.sync 11;" ::: "memory");
    asm volatile("{ .reg .pred p, q; setp.ne.s32 p, %1, 0; "
                 "bar.red.and.pred q, 11, p; selp.u32 %0, 1, 0, q; }"
                 : "=r"(every) : "r"(vote) : "memory");
    asm volatile("{ .reg .pred p, q; setp.ne.s32 p, %1, 0; "
                 "bar.red.or.pred q, 11, p; selp.u32 %0, 1, 0, q; }"
                 : "=r"(any) : "r"(vote) : "memory");
    const unsigned base = ((block * 7 + round) * blockDim.x + tid) * 3;
    output[base] = count;
    output[base + 1] = every;
    output[base + 2] = any;
  }
}

// A whole-CTA reduction must release when the nonparticipants exit. This
// includes an exit inside a partial warp, leaving fewer than 32 live lanes.
__global__ void exiting(int* output) {
  const unsigned tid = threadIdx.x;
  if (tid >= 17) return;
  unsigned count;
  asm volatile("{ .reg .pred p; setp.eq.u32 p, %1, %1; "
               "bar.red.popc.u32 %0, 14, p; }"
               : "=r"(count) : "r"(tid) : "memory");
  asm volatile("bar.sync 14;" ::: "memory");
  output[blockIdx.x * 17 + tid] = count;
}

// Each independent pair has its own producer/consumer barrier and buffer.
// The acknowledgement on a different barrier prevents early reuse.
__global__ void pairs(int* output) {
  __shared__ int data[64];
  const unsigned tid = threadIdx.x, lane = tid % 32;
  const unsigned pair = tid / 64, role = (tid / 32) % 2;
  const unsigned filled = pair * 2 + 1, consumed = filled + 1;
  for (unsigned round = 0; round < 7; ++round) {
    if (role == 0) {
      data[pair * 32 + lane] = blockIdx.x * 10000 + pair * 1000 + round * 100 + lane;
      asm volatile("bar.arrive %0, 64;" :: "r"(filled) : "memory");
      asm volatile("bar.sync %0, 64;" :: "r"(consumed) : "memory");
    } else {
      asm volatile("bar.sync %0, 64;" :: "r"(filled) : "memory");
      const unsigned base = (blockIdx.x * 7 + round) * 64 + pair * 32 + lane;
      output[base] = data[pair * 32 + lane];
      asm volatile("bar.arrive %0, 64;" :: "r"(consumed) : "memory");
    }
  }
}

int main() {
  int* device;
  constexpr unsigned blocks = 5, rounds = 7;
  constexpr unsigned max_values = blocks * rounds * 96 * 3;
  int output[max_values];
  CUDA(cudaMalloc(&device, sizeof(output)));
  unsigned comparisons = 0;
  for (unsigned threads : {1u, 17u, 31u, 32u, 33u, 40u, 63u, 64u, 65u, 96u}) {
    reductions<<<blocks, threads>>>(device);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(output, device, blocks * rounds * threads * 3 * sizeof(int), cudaMemcpyDeviceToHost));
    for (unsigned b = 0; b < blocks; ++b) for (unsigned r = 0; r < rounds; ++r) {
      unsigned expected = 0;
      for (unsigned t = 0; t < threads; ++t) expected += ((t + b + r) % 5) < 2;
      for (unsigned t = 0; t < threads; ++t) {
        unsigned base = ((b * rounds + r) * threads + t) * 3;
        if (output[base] != int(expected) || output[base + 1] != int(expected == threads) ||
            output[base + 2] != int(expected != 0)) return 1;
        comparisons += 3;
      }
    }
  }
  for (unsigned threads : {17u, 31u, 32u, 33u, 40u, 64u, 65u, 96u}) {
    exiting<<<blocks, threads>>>(device);
    CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(output, device, blocks * 17 * sizeof(int), cudaMemcpyDeviceToHost));
    for (unsigned i = 0; i < blocks * 17; ++i) {
      if (output[i] != 17) return 2;
      ++comparisons;
    }
  }
  pairs<<<blocks, 128>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, device, blocks * rounds * 64 * sizeof(int), cudaMemcpyDeviceToHost));
  for (unsigned b = 0; b < blocks; ++b) for (unsigned r = 0; r < rounds; ++r)
    for (unsigned p = 0; p < 2; ++p) for (unsigned lane = 0; lane < 32; ++lane) {
      if (output[(b * rounds + r) * 64 + p * 32 + lane] != int(b * 10000 + p * 1000 + r * 100 + lane)) return 3;
      ++comparisons;
    }
  CUDA(cudaFree(device));
  std::printf("PASS cta_barrier_phases comparisons=%u launches=19\n", comparisons);
}
