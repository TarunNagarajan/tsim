#include "common.cuh"
#include <vector>
constexpr int tile = 8;
__global__ void tiled_matmul(const float* a, const float* b, float* c, int n) {
  __shared__ float as[tile][tile], bs[tile][tile];
  int x = threadIdx.x, y = threadIdx.y;
  int row = blockIdx.y * tile + y, col = blockIdx.x * tile + x;
  float sum = 0;
  for (int base = 0; base < n; base += tile) {
    as[y][x] = row < n && base + x < n ? a[row * n + base + x] : 0;
    bs[y][x] = col < n && base + y < n ? b[(base + y) * n + col] : 0;
    __syncthreads();
    for (int k = 0; k < tile; ++k) sum += as[y][k] * bs[k][x];
    __syncthreads();
  }
  if (row < n && col < n) c[row * n + col] = sum;
}
int main() {
  constexpr int n = 19;
  std::vector<float> a(n * n), b(n * n), c(n * n);
  for (int i = 0; i < n * n; ++i) { a[i] = float(i % 7 - 3); b[i] = float(i % 5 - 2); }
  float *da, *db, *dc; size_t bytes = n * n * sizeof(float);
  CUDA(cudaMalloc(&da, bytes)); CUDA(cudaMalloc(&db, bytes)); CUDA(cudaMalloc(&dc, bytes));
  CUDA(cudaMemcpy(da, a.data(), bytes, cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db, b.data(), bytes, cudaMemcpyHostToDevice));
  tiled_matmul<<<dim3((n + tile - 1) / tile, (n + tile - 1) / tile), dim3(tile, tile)>>>(da, db, dc, n);
  CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(c.data(), dc, bytes, cudaMemcpyDeviceToHost));
  for (int row = 0; row < n; ++row) for (int col = 0; col < n; ++col) {
    float reference = 0;
    for (int k = 0; k < n; ++k) reference += a[row * n + k] * b[k * n + col];
    if (c[row * n + col] != reference) return 1;
  }
  CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
  std::printf("PASS tiled_matmul n=%d first=%.0f last=%.0f\n", n, c.front(), c.back());
}
