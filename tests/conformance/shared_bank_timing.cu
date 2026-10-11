#include "../../examples/common.cuh"
#include <vector>

constexpr int blocks=3,width=64,fields=7;

__global__ void bank_traffic(const unsigned* input,unsigned long long* output) {
  __shared__ unsigned strided[width*32];
  __shared__ __align__(16) unsigned vectors[width*4];
  __shared__ unsigned char bytes[width];
  __shared__ unsigned short halves[width];
  __shared__ unsigned long long wide[width],counter64;
  __shared__ unsigned counter32,broadcast;
  unsigned t=threadIdx.x,base=blockIdx.x*width;
  if(t==0) {counter32=0;counter64=0;broadcast=900+blockIdx.x;}
  strided[t*32]=base+t+11;
  wide[t]=10000000000ULL+base+t;
  unsigned p=static_cast<unsigned>(__cvta_generic_to_shared(vectors+t*4));
  unsigned a=t+1,b=t+2,c=t+3,d=t+4;
  asm volatile("st.shared.v4.u32 [%0], {%1,%2,%3,%4};"::"r"(p),"r"(a),"r"(b),"r"(c),"r"(d):"memory");
  unsigned bp=static_cast<unsigned>(__cvta_generic_to_shared(bytes+t));
  unsigned hp=static_cast<unsigned>(__cvta_generic_to_shared(halves+t));
  asm volatile("st.shared.u8 [%0], %1;"::"r"(bp),"r"(t):"memory");
  unsigned hv=1000+t;
  asm volatile("st.shared.u16 [%0], %1;"::"r"(hp),"r"(hv):"memory");
  __syncthreads();
  asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];":"=r"(a),"=r"(b),"=r"(c),"=r"(d):"r"(p):"memory");
  unsigned bv,half;
  asm volatile("ld.shared.u8 %0, [%1];":"=r"(bv):"r"(bp):"memory");
  asm volatile("ld.shared.u16 %0, [%1];":"=r"(half):"r"(hp):"memory");
  const unsigned* source=(t&1)?input+base+t:strided+t*32;
  unsigned generic;
  asm volatile("ld.u32 %0, [%1];":"=r"(generic):"l"(source):"memory");
  unsigned old32;unsigned long long old64;
  unsigned cp=static_cast<unsigned>(__cvta_generic_to_shared(&counter32));
  unsigned c64p=static_cast<unsigned>(__cvta_generic_to_shared(&counter64));
  asm volatile("atom.shared.add.u32 %0, [%1], 1;":"=r"(old32):"r"(cp):"memory");
  asm volatile("atom.shared.add.u64 %0, [%1], 2;":"=l"(old64):"r"(c64p):"memory");
  // Every warp's shared updates precede this actual CTA handoff.
  __syncthreads();
  unsigned long long* dst=output+(base+t)*fields;
  dst[0]=a+b+c+d;dst[1]=bv+half;dst[2]=generic;dst[3]=wide[t];
  dst[4]=broadcast;dst[5]=counter32;dst[6]=counter64;
}

int main() {
  std::vector<unsigned> input(blocks*width);
  for(unsigned t=0;t<input.size();++t)input[t]=7000+t;
  unsigned* device_input;unsigned long long* device_output;
  CUDA(cudaMalloc(&device_input,input.size()*sizeof(unsigned)));
  CUDA(cudaMalloc(&device_output,input.size()*fields*sizeof(unsigned long long)));
  CUDA(cudaMemcpy(device_input,input.data(),input.size()*sizeof(unsigned),cudaMemcpyHostToDevice));
  std::vector<unsigned long long> got(input.size()*fields);
  for(int round=0;round<2;++round) {
    bank_traffic<<<blocks,width>>>(device_input,device_output);
    CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(got.data(),device_output,got.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    for(unsigned t=0;t<input.size();++t) {
      unsigned lane=t%width;
      unsigned long long expected[fields]={4*lane+10,1000+2*lane,(lane&1)?7000+t:t+11,
        10000000000ULL+t,900+t/width,width,2*width};
      for(int f=0;f<fields;++f)if(got[t*fields+f]!=expected[f]) {
        std::fprintf(stderr,"shared_bank_timing mismatch round=%d thread=%u field=%d got=%llu expected=%llu\n",round,t,f,got[t*fields+f],expected[f]);return 1;
      }
    }
  }
  CUDA(cudaFree(device_input));CUDA(cudaFree(device_output));
  std::puts("PASS shared_bank_timing values=2688 launches=2 shared=vector4/byte/half/64/broadcast/atomic32/atomic64 mixed=global/shared");
}
