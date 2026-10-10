#include "common.cuh"
__global__ void warp_atomic(int* total, unsigned* votes) {
  int lane = threadIdx.x & 31;
  int value = lane + 1;
  unsigned mask = 0xffffffffu;
  unsigned vote = __ballot_sync(mask, lane < 16);
  for (int offset = 16; offset; offset /= 2)
    value += __shfl_down_sync(mask, value, offset);
  if (lane == 0) { atomicAdd(total, value); votes[threadIdx.x / 32] = vote; }
}
int main() {
  int* device; unsigned* dvotes; int total; unsigned votes[4];
  CUDA(cudaMalloc(&device, sizeof(int))); CUDA(cudaMalloc(&dvotes, sizeof(votes)));
  CUDA(cudaMemset(device, 0, sizeof(int)));
  warp_atomic<<<1, 128>>>(device, dvotes);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(&total, device, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA(cudaMemcpy(votes, dvotes, sizeof(votes), cudaMemcpyDeviceToHost));
  CUDA(cudaFree(device)); CUDA(cudaFree(dvotes));
  if (total != 4 * 32 * 33 / 2) return 1;
  for (unsigned vote : votes) if (vote != 0xffffu) return 1;
  std::printf("PASS warp_atomic total=%d ballot=%x\n", total, votes[0]);
}
