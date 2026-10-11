#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { std::fprintf(stderr,"%s\n",cudaGetErrorString(e)); std::exit(2); } } while(0)
constexpr int blocks=5, width=64, rounds=3;

__global__ void handoffs(int* scratch,int* output) {
  int lane=threadIdx.x&31, warp=threadIdx.x>>5, base=blockIdx.x*32;
  for(int round=0;round<rounds;++round) {
    if(warp==0) {
      scratch[base+lane]=1000*round+100*blockIdx.x+lane;
      asm volatile("bar.arrive 1, 64;" ::: "memory");
      // This request follows arrival at ID 1 and belongs to the next phase's
      // prefix, not the already announced publication.
      output[blocks*32*rounds+round*blocks*32+base+lane]=7000+1000*round+100*blockIdx.x+lane;
      asm volatile("bar.sync 2, 64;" ::: "memory");
    } else {
      asm volatile("bar.sync 1, 64;" ::: "memory");
      output[round*blocks*32+base+lane]=scratch[base+lane];
      asm volatile("bar.arrive 2, 64;" ::: "memory");
    }
  }
}

__global__ void saved_reductions(int* output) {
  int thread=threadIdx.x, base=blockIdx.x*width;
  for(int round=0;round<rounds;++round) {
    unsigned input=((thread+round)%3)==0, popc, all, any;
    asm volatile("{ .reg .pred p; setp.ne.u32 p, %1, 0; bar.red.popc.u32 %0, 3, p; }" : "=r"(popc) : "r"(input) : "memory");
    asm volatile("{ .reg .pred p,q; setp.ne.u32 p, %1, 0; bar.red.and.pred q, 3, p; selp.u32 %0, 1, 0, q; }" : "=r"(all) : "r"(input) : "memory");
    asm volatile("{ .reg .pred p,q; setp.ne.u32 p, %1, 0; bar.red.or.pred q, 3, p; selp.u32 %0, 1, 0, q; }" : "=r"(any) : "r"(input) : "memory");
    output[(round*3+0)*blocks*width+base+thread]=popc;
    output[(round*3+1)*blocks*width+base+thread]=all;
    output[(round*3+2)*blocks*width+base+thread]=any;
  }
}

__global__ void exited_publication(int* scratch,int* output) {
  int lane=threadIdx.x&31, base=blockIdx.x*32;
  if(threadIdx.x>=32) { scratch[base+lane]=9000+100*blockIdx.x+lane; return; }
  asm volatile("bar.sync 4;" ::: "memory");
  output[base+lane]=scratch[base+lane];
}

int main() {
  int *scratch,*output;
  constexpr int size=blocks*width*rounds*3;
  int host[size], comparisons=0;
  CUDA(cudaMalloc(&scratch,blocks*32*sizeof(int))); CUDA(cudaMalloc(&output,sizeof(host)));
  CUDA(cudaMemset(scratch,0,blocks*32*sizeof(int))); CUDA(cudaMemset(output,0,sizeof(host)));
  handoffs<<<blocks,width>>>(scratch,output); CUDA(cudaDeviceSynchronize()); CUDA(cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost));
  for(int round=0;round<rounds;++round) for(int block=0;block<blocks;++block) for(int lane=0;lane<32;++lane) {
    int value=1000*round+100*block+lane, index=round*blocks*32+block*32+lane;
    if(host[index]!=value || host[blocks*32*rounds+index]!=7000+value) return 3;
    comparisons+=2;
  }
  saved_reductions<<<blocks,width>>>(output); CUDA(cudaDeviceSynchronize()); CUDA(cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost));
  for(int round=0;round<rounds;++round) {
    int count=0; for(int thread=0;thread<width;++thread) count+=((thread+round)%3)==0;
    for(int block=0;block<blocks;++block) for(int thread=0;thread<width;++thread) {
      int index=block*width+thread;
      if(host[(round*3+0)*blocks*width+index]!=count || host[(round*3+1)*blocks*width+index]!=0 || host[(round*3+2)*blocks*width+index]!=1) return 4;
      comparisons+=3;
    }
  }
  exited_publication<<<blocks,width>>>(scratch,output); CUDA(cudaDeviceSynchronize()); CUDA(cudaMemcpy(host,output,sizeof(host),cudaMemcpyDeviceToHost));
  for(int block=0;block<blocks;++block) for(int lane=0;lane<32;++lane) {
    if(host[block*32+lane]!=9000+100*block+lane) return 5;
    ++comparisons;
  }
  CUDA(cudaFree(scratch)); CUDA(cudaFree(output));
  std::printf("PASS cta_timing comparisons=%d launches=3\n",comparisons);
}
