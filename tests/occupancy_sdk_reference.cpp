// Calls the installed CUDA occupancy calculator on a CPU. No CUDA runtime,
// driver, GPU or SDK implementation copied into this repository is required.
#include <cuda_occupancy.h>
#include <cstdio>
int main() {
  int major, threads, registers, shared, dynamic;
  while (std::scanf("%d %d %d %d %d", &major, &threads, &registers, &shared, &dynamic) == 5) {
    cudaOccDeviceProp device;
    device.computeMajor = major; device.computeMinor = 0;
    device.maxThreadsPerBlock = 1024; device.maxThreadsPerMultiprocessor = 2048;
    device.regsPerBlock = device.regsPerMultiprocessor = 65536; device.warpSize = 32;
    device.sharedMemPerBlock = 48 * 1024;
    device.sharedMemPerMultiprocessor = (major == 8 ? 164 : 228) * 1024;
    device.sharedMemPerBlockOptin = device.sharedMemPerMultiprocessor - 1024;
    device.reservedSharedMemPerBlock = 1024; device.numSms = 1;
    cudaOccFuncAttributes function;
    function.maxThreadsPerBlock = 1024; function.numRegs = registers;
    function.sharedSizeBytes = shared; function.numBlockBarriers = 1;
    function.shmemLimitConfig = FUNC_SHMEM_LIMIT_OPTIN;
    function.maxDynamicSharedSizeBytes = device.sharedMemPerBlockOptin - shared;
    cudaOccDeviceState state; state.carveoutConfig = SHAREDMEM_CARVEOUT_MAX_SHARED;
    cudaOccResult result{};
    auto status = cudaOccMaxActiveBlocksPerMultiprocessor(&result, &device, &function, &state, threads, dynamic);
    if (status != CUDA_OCC_SUCCESS) return 2;
    std::printf("%d %d %zu\n", result.activeBlocksPerMultiprocessor,
                result.allocatedRegistersPerBlock, result.allocatedSharedMemPerBlock);
  }
}
