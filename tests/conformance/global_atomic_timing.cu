#include "../../examples/common.cuh"
#include <algorithm>
#include <vector>

constexpr unsigned blocks=3,width=64,count=blocks*width,fields=6;

__global__ void updates(unsigned* hot,unsigned long long* wide,unsigned* per_word,unsigned* scratch,unsigned long long* output) {
  __shared__ unsigned shared_counter;
  unsigned t=threadIdx.x,tid=blockIdx.x*width+t;
  if(t==0)shared_counter=0;
  __syncthreads();
  unsigned old;unsigned long long old64;
  asm volatile("atom.global.add.u32 %0,[%1],1;":"=r"(old):"l"(hot):"memory");
  asm volatile("atom.global.add.u64 %0,[%1],2;":"=l"(old64):"l"(wide):"memory");
  unsigned per;
  asm volatile("atom.global.add.u32 %0,[%1],3;":"=r"(per):"l"(per_word+(tid&31)):"memory");
  // One generic warp instruction routes odd lanes globally and even lanes to
  // the CTA's shared counter. Returned values must remain distinct RMW results.
  unsigned* mixed=(t&1)?hot+1:&shared_counter;
  unsigned mix;
  asm volatile("atom.add.u32 %0,[%1],1;":"=r"(mix):"l"(mixed):"memory");
  asm volatile("red.global.add.u32 [%0],5;"::"l"(hot+2):"memory");
  scratch[tid]=1000+tid;
  __syncthreads();
  unsigned long long* dst=output+tid*fields;
  dst[0]=old;dst[1]=old64;dst[2]=per;dst[3]=mix;
  dst[4]=shared_counter;dst[5]=scratch[blockIdx.x*width+((t+1)%width)];
}

__global__ void variants(unsigned* data,unsigned long long* output) {
  unsigned tid=blockIdx.x*width+threadIdx.x,initial=17+tid,value=900+tid,r;
  unsigned* address=data+tid;unsigned long long* dst=output+tid*11;
  asm volatile("atom.global.cas.b32 %0,[%1],%2,%3;":"=r"(r):"l"(address),"r"(initial),"r"(value):"memory");dst[0]=r;
  asm volatile("atom.global.cas.b32 %0,[%1],%2,9;":"=r"(r):"l"(address),"r"(initial):"memory");dst[1]=r;
  unsigned high=value+7,low=value+2;
  asm volatile("atom.global.max.u32 %0,[%1],%2;":"=r"(r):"l"(address),"r"(high):"memory");dst[2]=r;
  asm volatile("atom.global.min.u32 %0,[%1],%2;":"=r"(r):"l"(address),"r"(low):"memory");dst[3]=r;
  asm volatile("atom.global.exch.b32 %0,[%1],42;":"=r"(r):"l"(address):"memory");dst[4]=r;
  asm volatile("atom.global.and.b32 %0,[%1],15;":"=r"(r):"l"(address):"memory");dst[5]=r;
  asm volatile("atom.global.or.b32 %0,[%1],64;":"=r"(r):"l"(address):"memory");dst[6]=r;
  asm volatile("atom.global.xor.b32 %0,[%1],3;":"=r"(r):"l"(address):"memory");dst[7]=r;
  asm volatile("atom.global.inc.u32 %0,[%1],7;":"=r"(r):"l"(address):"memory");dst[8]=r;
  asm volatile("atom.global.dec.u32 %0,[%1],7;":"=r"(r):"l"(address):"memory");dst[9]=r;
  asm volatile("red.global.min.u32 [%0],2;"::"l"(address):"memory");dst[10]=*address;
}

int main() {
  unsigned *hot,*per,*scratch,*data;unsigned long long *wide,*out;
  CUDA(cudaMalloc(&hot,3*sizeof(unsigned)));CUDA(cudaMalloc(&wide,sizeof(unsigned long long)));
  CUDA(cudaMalloc(&per,32*sizeof(unsigned)));CUDA(cudaMalloc(&scratch,count*sizeof(unsigned)));
  CUDA(cudaMalloc(&out,count*11*sizeof(unsigned long long)));CUDA(cudaMalloc(&data,count*sizeof(unsigned)));
  std::vector<unsigned long long> got(count*fields),tickets(count),wide_tickets(count);
  unsigned totals[3],words[32];unsigned long long sum64;
  unsigned comparisons=0;
  auto check=[&](unsigned long long actual,unsigned long long expected) {
    ++comparisons;if(actual!=expected) {std::fprintf(stderr,"global_atomic_timing mismatch comparison=%u got=%llu expected=%llu\n",comparisons,actual,expected);std::exit(1);}
  };
  for(unsigned round=0;round<2;++round) {
    CUDA(cudaMemset(hot,0,3*sizeof(unsigned)));CUDA(cudaMemset(wide,0,sizeof(unsigned long long)));CUDA(cudaMemset(per,0,32*sizeof(unsigned)));
    updates<<<blocks,width>>>(hot,wide,per,scratch,out);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    CUDA(cudaMemcpy(got.data(),out,got.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(totals,hot,sizeof(totals),cudaMemcpyDeviceToHost));CUDA(cudaMemcpy(words,per,sizeof(words),cudaMemcpyDeviceToHost));
    CUDA(cudaMemcpy(&sum64,wide,sizeof(sum64),cudaMemcpyDeviceToHost));
    check(totals[0],count);check(totals[1],count/2);check(totals[2],5*count);check(sum64,2*count);
    for(unsigned word=0;word<32;++word)check(words[word],3*(count/32));
    for(unsigned tid=0;tid<count;++tid) {
      tickets[tid]=got[tid*fields];wide_tickets[tid]=got[tid*fields+1];
      check(got[tid*fields+4],width/2);check(got[tid*fields+5],1000+(tid/width)*width+((tid+1)%width));
    }
    std::sort(tickets.begin(),tickets.end());std::sort(wide_tickets.begin(),wide_tickets.end());
    for(unsigned t=0;t<count;++t){check(tickets[t],t);check(wide_tickets[t],2*t);}
    for(unsigned word=0;word<32;++word) {
      std::vector<unsigned long long> values;
      for(unsigned tid=word;tid<count;tid+=32)values.push_back(got[tid*fields+2]);
      std::sort(values.begin(),values.end());
      for(unsigned i=0;i<values.size();++i)check(values[i],3*i);
    }
    std::vector<unsigned long long> odd;
    for(unsigned b=0;b<blocks;++b) {
      std::vector<unsigned long long> even;
      for(unsigned t=0;t<width;++t)if(t&1)odd.push_back(got[(b*width+t)*fields+3]);else even.push_back(got[(b*width+t)*fields+3]);
      std::sort(even.begin(),even.end());for(unsigned i=0;i<even.size();++i)check(even[i],i);
    }
    std::sort(odd.begin(),odd.end());for(unsigned i=0;i<odd.size();++i)check(odd[i],i);
    std::vector<unsigned> initial(count);for(unsigned t=0;t<count;++t)initial[t]=17+t;
    CUDA(cudaMemcpy(data,initial.data(),count*sizeof(unsigned),cudaMemcpyHostToDevice));
    variants<<<blocks,width>>>(data,out);CUDA(cudaGetLastError());CUDA(cudaDeviceSynchronize());
    std::vector<unsigned long long> results(count*11);CUDA(cudaMemcpy(results.data(),out,results.size()*sizeof(unsigned long long),cudaMemcpyDeviceToHost));
    for(unsigned t=0;t<count;++t) {
      unsigned long long expected[11]={17+t,900+t,900+t,907+t,902+t,42,10,74,73,0,2};
      for(unsigned f=0;f<11;++f)check(results[t*11+f],expected[f]);
    }
  }
  CUDA(cudaFree(hot));CUDA(cudaFree(wide));CUDA(cudaFree(per));CUDA(cudaFree(scratch));CUDA(cudaFree(out));CUDA(cudaFree(data));
  std::printf("PASS global_atomic_timing comparisons=%u launches=4 atomic32/atomic64/per_word/mixed/red/CTA/cas/min/max/exch/bit/inc/dec\n",comparisons);
}
