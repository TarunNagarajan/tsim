#include "../../examples/common.cuh"
#include <vector>

struct Result { unsigned values[29]; };
__global__ void collectives(Result* output) {
  unsigned lane = threadIdx.x & 31;
  unsigned warp = threadIdx.x >> 5;
  unsigned value = warp * 100 + lane;
  Result r{};
  int slot = 0;
  for (int width = 2; width <= 32; width *= 2) {
    r.values[slot++] = __shfl_sync(0xffffffffu, value, 1, width);
    r.values[slot++] = __shfl_up_sync(0xffffffffu, value, 1, width);
    r.values[slot++] = __shfl_down_sync(0xffffffffu, value, 1, width);
    r.values[slot++] = __shfl_xor_sync(0xffffffffu, value, 1, width);
  }
  r.values[slot++] = __ballot_sync(0xffffffffu, lane % 3 == 0);
  r.values[slot++] = __all_sync(0xffffffffu, lane < 32);
  r.values[slot++] = __any_sync(0xffffffffu, lane == 17);
  r.values[slot++] = __all_sync(0xffffffffu, lane != 17);
  r.values[slot++] = __any_sync(0xffffffffu, false);
  r.values[slot++] = __match_any_sync(0xffffffffu, lane % 4);
  int all = 0;
  r.values[slot++] = __match_all_sync(0xffffffffu, 7u, &all);
  r.values[slot++] = all;
  r.values[slot++] = __reduce_add_sync(0xffffffffu, lane + 1);
  output[threadIdx.x] = r;
}
int main() {
  Result* device;
  CUDA(cudaMalloc(&device, 64 * sizeof(Result)));
  collectives<<<1, 64>>>(device);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<Result> got(64);
  CUDA(cudaMemcpy(got.data(), device, 64 * sizeof(Result), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device));
  unsigned ballot = 0;
  for (unsigned lane = 0; lane < 32; ++lane) if (lane % 3 == 0) ballot |= 1u << lane;
  for (unsigned tid = 0; tid < 64; ++tid) {
    unsigned lane = tid % 32, warp = tid / 32, slot = 0;
    Result expected{};
    for (unsigned width = 2; width <= 32; width *= 2) {
      unsigned base = lane / width * width;
      expected.values[slot++] = warp * 100 + base + 1;
      expected.values[slot++] = warp * 100 + (lane % width ? lane - 1 : lane);
      expected.values[slot++] = warp * 100 + (lane % width + 1 < width ? lane + 1 : lane);
      expected.values[slot++] = warp * 100 + (lane ^ 1);
    }
    unsigned match = 0;
    for (unsigned other = 0; other < 32; ++other) if (other % 4 == lane % 4) match |= 1u << other;
    unsigned tail[] = {ballot, 1, 1, 0, 0, match, 0xffffffffu, 1, 528};
    for (unsigned value : tail) expected.values[slot++] = value;
    for (unsigned i = 0; i < 29; ++i) if (got[tid].values[i] != expected.values[i]) {
      std::fprintf(stderr, "warp_collectives mismatch tid=%u field=%u got=%u expected=%u\n", tid, i, got[tid].values[i], expected.values[i]);
      return 1;
    }
  }
  std::puts("PASS warp_collectives threads=64 fields=29");
}
