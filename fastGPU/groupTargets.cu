#include "Treecode.h"

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/scan.h>

namespace groupTargets
{

  template<typename T>
  static __global__ void shuffle(const int n, const int *map, const T *in, T *out)
  {
    const int gidx = blockDim.x*blockIdx.x + threadIdx.x;
    if (gidx >= n) return;

    out[gidx] = in[map[gidx]];
  }

  template<int NBITS>
  static __device__ unsigned long long get_key(int3 crd)
  {
    int i,xi, yi, zi;
    int mask;
    unsigned long long key;

    //0= 000, 1=001, 2=011, 3=010, 4=110, 5=111, 6=101, 7=100
    //000=0=0, 001=1=1, 011=3=2, 010=2=3, 110=6=4, 111=7=5, 101=5=6, 100=4=7
    const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};

    int temp;

    mask = 1 << (NBITS - 1);
    key  = 0;

#if 0
    uint4 key_new;
#endif

#pragma unroll
    for(i = 0; i < NBITS; i++, mask >>= 1)
      {
        xi = (crd.x & mask) ? 1 : 0;
        yi = (crd.y & mask) ? 1 : 0;
        zi = (crd.z & mask) ? 1 : 0;        

        const int index = (xi << 2) + (yi << 1) + zi;

        int Cvalue;
        if(index == 0)
	  {
	    temp = crd.z; crd.z = crd.y; crd.y = temp;
	    Cvalue = C[0];
	  }
        else  if(index == 1 || index == 5)
	  {
	    temp = crd.x; crd.x = crd.y; crd.y = temp;
	    if (index == 1) Cvalue = C[1];
	    else            Cvalue = C[5];
	  }
        else  if(index == 4 || index == 6)
	  {
	    crd.x = (crd.x) ^ (-1);
	    crd.z = (crd.z) ^ (-1);
	    if (index == 4) Cvalue = C[4];
	    else            Cvalue = C[6];
	  }
        else  if(index == 7 || index == 3)
	  {
	    temp  = (crd.x) ^ (-1);         
	    crd.x = (crd.y) ^ (-1);
	    crd.y = temp;
	    if (index == 3) Cvalue = C[3];
	    else            Cvalue = C[7];
	  }
        else
	  {
	    temp = (crd.z) ^ (-1);         
	    crd.z = (crd.y) ^ (-1);
	    crd.y = temp;          
	    Cvalue = C[2];
	  }   

        key = (key<<3) + Cvalue;

#if 0
        if(i == 19)
	  {
	    key_new.y = key;
	    key = 0;
	  }
        if(i == 9)
	  {
	    key_new.x = key;
	    key = 0;
	  }
#endif
      } //end for

#if 0
    key_new.z = key;

    return key_new;
#else
    return key;
#endif
  }

  template<int NBINS>
  static __global__ 
  void computeKeys(
		   const int n,
		   const float4 *d_domain,
		   const float4 *bodyPos,
		   unsigned long long *keys,
		   int *values)
  {
    const int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float4 body = bodyPos[idx];

    const float4 domain = d_domain[0];
    const float inv_domain_size = 0.5f / domain.w;
    const float3 bmin = {domain.x - domain.w,
			 domain.y - domain.w,
			 domain.z - domain.w};

    const int xc = static_cast<int>((body.x - bmin.x) * inv_domain_size * (1<<NBINS));
    const int yc = static_cast<int>((body.y - bmin.y) * inv_domain_size * (1<<NBINS));
    const int zc = static_cast<int>((body.z - bmin.z) * inv_domain_size * (1<<NBINS));

    keys  [idx] = get_key<NBINS>(make_int3(xc,yc,zc));
    values[idx] = idx;
  }

  static __global__
  void mask_keys(
		 const int n, 
		 const unsigned long long mask,
		 unsigned long long *keys,
		 unsigned long long *keys_inv,
		 int *bodyBegIdx,
		 int *bodyEndIdx)
  {
    const int gidx = blockIdx.x*blockDim.x + threadIdx.x;
    if (gidx >= n) return;

#if 0
    if (gidx < 100)
      {
        printf("gidx= %d : keys= %llx  maks= %llx  res= %llx\n",
	       gidx, keys[gidx], mask, keys[gidx] & mask);
      }

#endif
    keys[gidx] &= mask;
    keys_inv[n-gidx-1] = keys[gidx];

    extern __shared__ unsigned long long shKeys[];

    const int tid = threadIdx.x;
    shKeys[tid+1] = keys[gidx] & mask;

    int shIdx = 0;
    int gmIdx = max(blockIdx.x*blockDim.x-1,0);
    if (tid == 1)
      {
        shIdx = blockDim.x+1;
        gmIdx = min(blockIdx.x*blockDim.x + blockDim.x,n-1);
      }
    if (tid < 2)
      shKeys[shIdx] = keys[gmIdx] & mask;

    __syncthreads();

    const int idx = tid+1;
    const unsigned long long currKey = shKeys[idx  ];
    const unsigned long long prevKey = shKeys[idx-1];
    const unsigned long long nextKey = shKeys[idx+1];

    if (currKey != prevKey || gidx == 0)
      bodyBegIdx[gidx] = gidx;
    else
      bodyBegIdx[gidx] = 0;

    if (currKey != nextKey || gidx == n-1)
      bodyEndIdx[n-1-gidx] = gidx+1;
    else
      bodyEndIdx[n-1-gidx] = 0;

  }

  __device__ unsigned int groupCounter= 0;

  static __global__
  void make_groups(const int n, const int nCrit,
		   const int *bodyBegIdx, 
		   const int *bodyEndIdx,
		   int2 *targetCells)
  {
    const int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gidx >= n) return;

    const int bodyBeg = bodyBegIdx[gidx];
    assert(gidx >= bodyBeg);

    const int igroup   = (gidx - bodyBeg)/nCrit;
    const int groupBeg = bodyBeg + igroup * nCrit;

#if 0
    if (gidx < 100)
      printf("gidx= %d  groupBeg =%d\n",gidx, groupBeg);
    return;
#endif

    if (gidx == groupBeg)
      {
        const int groupIdx = atomicAdd(&groupCounter,1);
        const int bodyEnd = bodyEndIdx[n-1-gidx];
        targetCells[groupIdx] = make_int2(groupBeg, min(nCrit, bodyEnd - groupBeg));
      }
  }

  struct keyCompare
  {
    __host__ __device__
    bool operator()(const unsigned long long x, const unsigned long long y)
    {
      return x < y;
    }
  };

};

void Treecode::groupTargets(int levelSplit, const int nCrit)
{
  this->nCrit = nCrit;
  const int nthread = 256;

  d_key.realloc(2.0*nBody);
  d_value.realloc(nBody);
  d_targetCells.realloc(nBody);

  unsigned long long *d_keys = (unsigned long long*)d_key.ptr;
  int *d_values = d_value.ptr;

  numTargets = 0;
  CUDA_SAFE_CALL(cudaMemcpyToSymbol(groupTargets::groupCounter, &numTargets, sizeof(int)));

  const int nblock  = (nBody-1)/nthread + 1;
  const int NBINS = 21; 

  cudaDeviceSynchronize();
  const double t0 = get_time();
  groupTargets::computeKeys<NBINS><<<nblock,nthread>>>(nBody, d_domain, d_bodyPos, d_keys, d_values);

  levelSplit = std::max(1,levelSplit);  /* pick the coarse segment boundaries at the levelSplit */
  unsigned long long mask= 0;
  for (int i = 0; i < NBINS; i++)
    {
      mask <<= 3;
      if (i < levelSplit)
	mask |= 0x7;
    }
  //printf("mask= %llx  \n", mask);

  /* sort particles by PH key */
  thrust::device_ptr<unsigned long long> keys_beg(d_keys);
  thrust::device_ptr<unsigned long long> keys_end(d_keys + nBody);
  thrust::device_ptr<int> vals_beg(d_value.ptr);
#if 1
  thrust::sort_by_key(keys_beg, keys_end, vals_beg); 
#else
  thrust::sort_by_key(keys_beg, keys_end, vals_beg, groupTargets::keyCompare());
#endif

#if 1
  groupTargets::shuffle<float4><<<nblock,nthread>>>(nBody, d_value, d_bodyPos, d_bodyPos_tmp);

  cuda_mem<int> d_bodyBegIdx, d_bodyEndIdx;
  cuda_mem<unsigned long long> d_keys_inv;
  d_bodyBegIdx.alloc(nBody);
  d_bodyEndIdx.alloc(nBody);
  d_keys_inv.alloc(nBody);
  groupTargets::mask_keys<<<nblock,nthread,(nthread+2)*sizeof(unsigned long long)>>>(nBody, mask, d_keys, d_keys_inv, d_bodyBegIdx, d_bodyEndIdx);

  thrust::device_ptr<int> valuesBeg(d_bodyBegIdx.ptr);
  thrust::device_ptr<int> valuesEnd(d_bodyEndIdx.ptr);
  thrust::inclusive_scan_by_key(keys_beg,     keys_end,    valuesBeg, valuesBeg);

  thrust::device_ptr<unsigned long long> keys_inv_beg(d_keys_inv.ptr);
  thrust::device_ptr<unsigned long long> keys_inv_end(d_keys_inv.ptr + nBody);
  thrust::inclusive_scan_by_key(keys_inv_beg, keys_inv_end, valuesEnd, valuesEnd);

#if 0
  std::vector<int> beg(nBody), end(nBody);
  std::vector<unsigned long long> h_keys(nBody);
  d_bodyBegIdx.d2h(&beg[0]);
  d_bodyEndIdx.d2h(&end[0]);
  d_key.d2h((int*)&h_keys[0],2*nBody);
  for (int i = 0; i < nBody; i++)
    {
      printf("i= %d : keys= %llx beg= %d  end= %d\n", i, h_keys[i], beg[i], end[nBody-1-i]);
    }
#endif

  groupTargets::make_groups<<<nblock,nthread>>>(nBody, nCrit, d_bodyBegIdx, d_bodyEndIdx, d_targetCells);
#endif

  kernelSuccess("groupTargets");
  const double dt = get_time() - t0;
  fprintf(stdout,"Make groups          : %.7f s\n", dt);
  CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numTargets, groupTargets::groupCounter, sizeof(int)));
#if 0
  assert(0);
#endif

  //fprintf(stderr, "nGroup= %d <nCrit>= %g \n", numTargets, nBody*1.0/numTargets);
#if 0
  {
    std::vector<int2> groups(numTargets);
    d_targetCells.d2h((int2*)&groups[0], numTargets);
    int np_in_group = 0;
    for (int i = 0 ;i < numTargets; i++)
      {
#if 0
	printf("groupdIdx= %d  :: pbeg= %d  np =%d \n", i, groups[i].x, groups[i].y);
#else
	np_in_group += groups[i].y;
#endif
      }
    printf("np_in_group= %d    np= %d\n", np_in_group, nBody);
    assert(0);
  }
#endif

}