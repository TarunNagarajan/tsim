#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
inline void check_cuda(cudaError_t result, const char* call, int line) {
  if (result != cudaSuccess) {
    std::fprintf(stderr, "CUDA error at line %d (%s): %s\n", line, call, cudaGetErrorString(result));
    std::exit(2);
  }
}
#define CUDA(call) check_cuda((call), #call, __LINE__)
