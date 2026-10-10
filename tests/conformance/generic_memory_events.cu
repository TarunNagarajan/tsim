#include "../../examples/common.cuh"
// One generic instruction reaches a different space in each group of lanes.
// Inline PTX prevents nvcc from specializing the access back to typed spaces.
__global__ void mixed_spaces(int* global, int* output) {
  __shared__ int shared[64];
  volatile int local[2];
  unsigned lane = threadIdx.x;
  int* pointer = lane % 3 == 0 ? shared + lane : lane % 3 == 1 ?
      const_cast<int*>(local) : global + lane;
  int value = 7 + lane, got;
  asm volatile("st.u32 [%0], %1;" :: "l"(pointer), "r"(value) : "memory");
  asm volatile("ld.u32 %0, [%1];" : "=r"(got) : "l"(pointer) : "memory");
  output[lane] = got;
}
__global__ void atomics(int* global, int* output) {
  __shared__ int count;
  if (threadIdx.x == 0) count = 0;
  __syncthreads();
  atomicAdd(global, 1);
  atomicAdd(&count, 1);
  __syncthreads();
  if (threadIdx.x == 0) output[0] = count;
}
int main() {
  int *global, *output, got[64];
  CUDA(cudaMalloc(&global, sizeof(got))); CUDA(cudaMalloc(&output, sizeof(got)));
  mixed_spaces<<<1, 64>>>(global, output);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(got, output, sizeof(got), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 64; ++i) if (got[i] != i + 7) return 1;
  CUDA(cudaMemset(global, 0, sizeof(got)));
  atomics<<<1, 64>>>(global, output);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  int sum, shared;
  CUDA(cudaMemcpy(&sum, global, 4, cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(&shared, output, 4, cudaMemcpyDeviceToHost));
  CUDA(cudaFree(global)); CUDA(cudaFree(output));
  if (sum != 64 || shared != 64) return 1;
  std::puts("PASS generic_memory_events mixed=64 atomics=128");
}
