#include "common.cuh"
#include <vector>
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  constexpr int n = 259; // Partial final warp and block exercise the bounds guard.
  std::vector<float> a(n), b(n), c(n);
  for (int i = 0; i < n; ++i) { a[i] = float(i); b[i] = float(2 * i + 1); }
  float *da, *db, *dc;
  CUDA(cudaMalloc(&da, n * sizeof(float))); CUDA(cudaMalloc(&db, n * sizeof(float)));
  CUDA(cudaMalloc(&dc, n * sizeof(float)));
  CUDA(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  vector_add<<<(n + 63) / 64, 64>>>(da, db, dc, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) if (c[i] != float(3 * i + 1)) return 1;
  CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
  std::printf("PASS vector_add n=%d last=%.0f\n", n, c.back());
}
