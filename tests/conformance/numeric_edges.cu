#include "../../examples/common.cuh"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

struct Result { unsigned hi, reversed, population, leading; int rounded[4]; float fused; };
__global__ void numeric(const float* inputs, Result* output, int n) {
  int tid = threadIdx.x;
  if (tid >= n) return;
  unsigned x = 0xf1234567u ^ (unsigned(tid) * 0x13579bdu);
  unsigned y = 0xdeadbeefu + unsigned(tid) * 97u;
  Result r;
  r.hi = __umulhi(x, y);
  r.reversed = __brev(x);
  r.population = __popc(x);
  r.leading = __clz(tid == 0 ? 0u : x);
  float value = inputs[tid];
  r.rounded[0] = __float2int_rn(value);
  r.rounded[1] = __float2int_rz(value);
  r.rounded[2] = __float2int_rd(value);
  r.rounded[3] = __float2int_ru(value);
  r.fused = __fmaf_rn(value, 1.00000011920928955078125f, -value);
  output[tid] = r;
}
int main() {
  float input[] = {0.0f, -0.0f, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f,
                   3.5f, -3.5f, 1.999f, -1.999f, 127.75f, -127.75f, 65535.5f, -65535.5f};
  constexpr int n = sizeof(input) / sizeof(input[0]);
  float* device_input; Result* device_output;
  CUDA(cudaMalloc(&device_input, sizeof(input))); CUDA(cudaMalloc(&device_output, n * sizeof(Result)));
  CUDA(cudaMemcpy(device_input, input, sizeof(input), cudaMemcpyHostToDevice));
  numeric<<<1, 32>>>(device_input, device_output, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  std::vector<Result> got(n);
  CUDA(cudaMemcpy(got.data(), device_output, n * sizeof(Result), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device_input)); CUDA(cudaFree(device_output));
  for (int tid = 0; tid < n; ++tid) {
    unsigned x = 0xf1234567u ^ (unsigned(tid) * 0x13579bdu), y = 0xdeadbeefu + unsigned(tid) * 97u;
    unsigned reversed = 0, population = 0;
    for (unsigned bit = 0; bit < 32; ++bit) { reversed |= ((x >> bit) & 1u) << (31 - bit); population += (x >> bit) & 1u; }
    unsigned leading = 32;
    if (tid != 0) for (unsigned bit = 0; bit < 32; ++bit) if (x & (1u << (31 - bit))) { leading = bit; break; }
    if (got[tid].hi != unsigned((uint64_t(x) * y) >> 32) || got[tid].reversed != reversed || got[tid].population != population || got[tid].leading != leading) return 1;
    int expected[] = {int(std::nearbyint(input[tid])), int(std::trunc(input[tid])), int(std::floor(input[tid])), int(std::ceil(input[tid]))};
    for (int i = 0; i < 4; ++i) if (got[tid].rounded[i] != expected[i]) {
      std::fprintf(stderr, "numeric_edges conversion mismatch tid=%d mode=%d got=%d expected=%d\n", tid, i, got[tid].rounded[i], expected[i]); return 1;
    }
    float fused = std::fma(input[tid], 1.00000011920928955078125f, -input[tid]);
    unsigned expected_bits, got_bits;
    std::memcpy(&expected_bits, &fused, 4); std::memcpy(&got_bits, &got[tid].fused, 4);
    if (got_bits != expected_bits) { std::fprintf(stderr, "numeric_edges FMA bits mismatch tid=%d got=%x expected=%x\n", tid, got_bits, expected_bits); return 1; }
  }
  std::puts("PASS numeric_edges inputs=16 checks=9");
}
