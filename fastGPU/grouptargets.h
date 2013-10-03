#pragma once

extern void sort(const int size, unsigned long long * key, int * value);
extern void scan(const int size, unsigned long long * key, int * value);

namespace {
  template<int NBITS>
    static __device__
    unsigned long long getHilbert(int3 pos) {
    const int C[8] = {0, 1, 7, 6, 3, 2, 4, 5};
    int temp;
    int mask = 1 << (NBITS - 1);
    unsigned long long key = 0;
#pragma unroll
    for (int i=0; i<NBITS; i++, mask >>= 1) {
      int ix = (pos.x & mask) ? 1 : 0;
      int iy = (pos.y & mask) ? 1 : 0;
      int iz = (pos.z & mask) ? 1 : 0;
      const int index = (ix << 2) + (iy << 1) + iz;

      int Cvalue;
      if(index == 0)
	{
	  temp = pos.z; pos.z = pos.y; pos.y = temp;
	  Cvalue = C[0];
	}
      else  if(index == 1 || index == 5)
	{
	  temp = pos.x; pos.x = pos.y; pos.y = temp;
	  if (index == 1) Cvalue = C[1];
	  else            Cvalue = C[5];
	}
      else  if(index == 4 || index == 6)
	{
	  pos.x = (pos.x) ^ (-1);
	  pos.z = (pos.z) ^ (-1);
	  if (index == 4) Cvalue = C[4];
	  else            Cvalue = C[6];
	}
      else  if(index == 7 || index == 3)
	{
	  temp  = (pos.x) ^ (-1);
	  pos.x = (pos.y) ^ (-1);
	  pos.y = temp;
	  if (index == 3) Cvalue = C[3];
	  else            Cvalue = C[7];
	}
      else
	{
	  temp = (pos.z) ^ (-1);
	  pos.z = (pos.y) ^ (-1);
	  pos.y = temp;
	  Cvalue = C[2];
	}

      key = (key<<3) + Cvalue;
    }
    return key;
  }

  template<int NBINS>
    static __global__
    void getKeys(const int n,
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

      keys  [idx] = getHilbert<NBINS>(make_int3(xc,yc,zc));
      values[idx] = idx;
    }

  static __global__
    void permuteBodies(const int n, const int *map, const float4 *in, float4 *out) {
    const int gidx = blockDim.x*blockIdx.x + threadIdx.x;
    if (gidx >= n) return;
    out[gidx] = in[map[gidx]];
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
    void make_groups(const int n, const int NCRIT,
		     const int *bodyBegIdx,
		     const int *bodyEndIdx,
		     int2 *targetRange)
  {
    const int gidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (gidx >= n) return;

    const int bodyBeg = bodyBegIdx[gidx];
    assert(gidx >= bodyBeg);

    const int igroup   = (gidx - bodyBeg)/NCRIT;
    const int groupBeg = bodyBeg + igroup * NCRIT;

    if (gidx == groupBeg)
      {
        const int groupIdx = atomicAdd(&groupCounter,1);
        const int bodyEnd = bodyEndIdx[n-1-gidx];
        targetRange[groupIdx] = make_int2(groupBeg, min(NCRIT, bodyEnd - groupBeg));
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
}

class Group {
 public:
  int targets(const int numBodies, float4 * d_bodyPos, float4 * d_bodyPos2,
	      float4 * d_domain, int2 * d_targetRange, int levelSplit, const int NCRIT) {
    const int NTHREAD = 256;
    const int NBLOCK = (numBodies-1) / NTHREAD + 1;
    const int NBINS = 21;
    cuda_mem<unsigned long long> d_key;
    cuda_mem<int> d_value;
    d_key.alloc(numBodies);
    d_value.alloc(numBodies);
    cudaDeviceSynchronize();

    const double t0 = get_time();
    getKeys<NBINS><<<NBLOCK,NTHREAD>>>(numBodies, d_domain, d_bodyPos, d_key.ptr, d_value.ptr);
    sort(numBodies, d_key.ptr, d_value.ptr);
    permuteBodies<<<NBLOCK,NTHREAD>>>(numBodies, d_value, d_bodyPos, d_bodyPos2);

    cuda_mem<int> d_bodyBegIdx, d_bodyEndIdx;
    cuda_mem<unsigned long long> d_key2;
    d_bodyBegIdx.alloc(numBodies);
    d_bodyEndIdx.alloc(numBodies);
    d_key2.alloc(numBodies);
    unsigned long long mask = 0;
    for (int i=0; i<NBINS; i++) {
      mask <<= 3;
      if (i < levelSplit)
	mask |= 0x7;
    }
    mask_keys<<<NBLOCK,NTHREAD,(NTHREAD+2)*sizeof(unsigned long long)>>>(numBodies, mask, d_key.ptr, d_key2.ptr, d_bodyBegIdx, d_bodyEndIdx);
    scan(numBodies, d_key.ptr, d_bodyBegIdx.ptr);
    scan(numBodies, d_key2.ptr, d_bodyEndIdx.ptr);
    make_groups<<<NBLOCK,NTHREAD>>>(numBodies, NCRIT, d_bodyBegIdx, d_bodyEndIdx, d_targetRange);
    kernelSuccess("groupTargets");
    const double dt = get_time() - t0;
    fprintf(stdout,"Make groups          : %.7f s\n", dt);
    int numTargets;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numTargets, groupCounter, sizeof(int)));
    return numTargets;
  }
};
