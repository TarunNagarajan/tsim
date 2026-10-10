#include <cuda_runtime.h>
#include <cstdio>

__global__ void fill(int* data, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = 3 * i + 7;
}
__global__ void transform(int* data, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) data[i] = data[i] * 5 - 2;
}
int main() {
    constexpr int count = 67;
    int* device = nullptr;
    int result[count] = {};
    if (cudaMalloc(&device, sizeof(result)) != cudaSuccess) return 1;
    fill<<<3, 32>>>(device, count);
    if (cudaDeviceSynchronize() != cudaSuccess) return 2;
    transform<<<3, 32>>>(device, count);
    if (cudaDeviceSynchronize() != cudaSuccess) return 3;
    if (cudaMemcpy(result, device, sizeof(result), cudaMemcpyDeviceToHost) != cudaSuccess) return 4;
    cudaFree(device);
    for (int i = 0; i < count; ++i) {
        int expected = 15 * i + 33;
        if (result[i] != expected) { std::fprintf(stderr, "Mismatch %d: %d vs %d\n", i, result[i], expected); return 5; }
    }
    std::printf("PASS execution_launches launches=2 count=%d last=%d\n", count, result[count - 1]);
    return 0;
}
