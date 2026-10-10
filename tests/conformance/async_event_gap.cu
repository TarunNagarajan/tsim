#include "../../examples/common.cuh"
__global__ void async_copy(const int* input, int* output) {
  __shared__ int shared[32];
  unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(shared + threadIdx.x));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" :: "r"(address), "l"(input + threadIdx.x));
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;" ::: "memory");
  __syncthreads();
  output[threadIdx.x] = shared[threadIdx.x];
}
int main() {
  int input[32], output[32], *a, *b;
  for (int i = 0; i < 32; ++i) input[i] = 3 * i + 1;
  CUDA(cudaMalloc(&a, sizeof(input))); CUDA(cudaMalloc(&b, sizeof(output)));
  CUDA(cudaMemcpy(a, input, sizeof(input), cudaMemcpyHostToDevice));
  async_copy<<<1, 32>>>(a, b);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(output, b, sizeof(output), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(a)); CUDA(cudaFree(b));
  for (int i = 0; i < 32; ++i) if (input[i] != output[i]) return 1;
  std::puts("PASS async_event_gap n=32");
}
