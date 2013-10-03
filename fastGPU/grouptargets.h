#pragma once

#define NBITS 21

extern void sort(const int size, unsigned long long * key, int * value);
extern void scan(const int size, unsigned long long * key, int * value);

namespace {
  __device__ unsigned int numTargetGlob= 0;

  static __device__
    unsigned long long getHilbert(int3 iX) {
    const int octantMap[8] = {0, 1, 7, 6, 3, 2, 4, 5};
    int mask = 1 << (NBITS - 1);
    unsigned long long key = 0;
#pragma unroll
    for (int i=0; i<NBITS; i++) {
      const int ix = (iX.x & mask) ? 1 : 0;
      const int iy = (iX.y & mask) ? 1 : 0;
      const int iz = (iX.z & mask) ? 1 : 0;
      const int octant = (ix << 2) + (iy << 1) + iz;
      if(octant == 0) {
	const int temp = iX.z;
	iX.z = iX.y;
	iX.y = temp;
      } else if(octant == 1 || octant == 5) {
	const int temp = iX.x;
	iX.x = iX.y;
	iX.y = temp;
      } else if(octant == 4 || octant == 6){
	iX.x = (iX.x) ^ (-1);
	iX.z = (iX.z) ^ (-1);
      } else if(octant == 3 || octant == 7) {
	const int temp = (iX.x) ^ (-1);
	iX.x = (iX.y) ^ (-1);
	iX.y = temp;
      } else {
	const int temp = (iX.z) ^ (-1);
	iX.z = (iX.y) ^ (-1);
	iX.y = temp;
      }
      key = (key<<3) + octantMap[octant];
      mask >>= 1;
    }
    return key;
  }

  static __global__
    void getKeys(const int numBodies,
		 const float4 * d_domain,
		 const float4 * bodyPos,
		 unsigned long long * keys,
		 int * values) {
    const int bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    const float4 pos = bodyPos[bodyIdx];
    const float4 domain = d_domain[0];
    const float diameter = 2 * domain.w / (1<<NBITS);
    const float3 Xmin = {domain.x - domain.w,
			 domain.y - domain.w,
			 domain.z - domain.w};
    const int ix = int((pos.x - Xmin.x) / diameter);
    const int iy = int((pos.y - Xmin.y) / diameter);
    const int iz = int((pos.z - Xmin.z) / diameter);
    keys[bodyIdx] = getHilbert(make_int3(ix, iy, iz));
    values[bodyIdx] = bodyIdx;
  }

  static __global__
    void permuteBodies(const int numBodies, const int * value,
		       const float4 * bodyPos, float4 * bodyPos2) {
    const int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    bodyPos2[bodyIdx] = bodyPos[value[bodyIdx]];
  }

  static __global__
    void maskKeys(const int numBodies,
		  const unsigned long long mask,
		  unsigned long long * keys,
		  unsigned long long * keys2,
		  int * bodyBeginIdx,
		  int * bodyEndIdx) {
    const int bodyIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    keys2[numBodies-bodyIdx-1] = keys[bodyIdx] & mask;
    const int nextBodyIdx = min(bodyIdx+1, numBodies-1);
    const int prevBodyIdx = max(bodyIdx-1, 0);
    const unsigned long long currKey = keys[bodyIdx] & mask;
    const unsigned long long nextKey = keys[nextBodyIdx] & mask;
    const unsigned long long prevKey = keys[prevBodyIdx] & mask;
    if (prevKey < currKey || bodyIdx == 0)
      bodyBeginIdx[bodyIdx] = bodyIdx;
    else
      bodyBeginIdx[bodyIdx] = 0;
    if (currKey < nextKey || bodyIdx == numBodies-1)
      bodyEndIdx[numBodies-1-bodyIdx] = bodyIdx+1;
    else
      bodyEndIdx[numBodies-1-bodyIdx] = 0;
  }

  static __global__
    void getTargetRange(const int numBodies,
		      const int * bodyBeginIdx,
		      const int * bodyEndIdx,
		      int2 * targetRange) {
    const int groupSize = WARP_SIZE * 2;
    const int bodyIdx = blockDim.x * blockIdx.x + threadIdx.x;
    if (bodyIdx >= numBodies) return;
    const int bodyBegin = bodyBeginIdx[bodyIdx];
    assert(bodyIdx >= bodyBegin);
    const int groupIdx = (bodyIdx - bodyBegin) / groupSize;
    const int groupBegin = bodyBegin + groupIdx * groupSize;
    if (bodyIdx == groupBegin) {
      const int targetIdx = atomicAdd(&numTargetGlob, 1);
      const int bodyEnd = bodyEndIdx[numBodies-1-bodyIdx];
      targetRange[targetIdx] = make_int2(groupBegin, min(groupSize, bodyEnd - groupBegin));
    }
  }
}

class Group {
 public:
  int targets(const int numBodies, float4 * d_bodyPos, float4 * d_bodyPos2,
	      float4 * d_domain, int2 * d_targetRange, int levelSplit) {
    const int NBLOCK = (numBodies-1) / NTHREAD + 1;
    cuda_mem<unsigned long long> d_key;
    cuda_mem<int> d_value;
    d_key.alloc(numBodies);
    d_value.alloc(numBodies);
    cudaDeviceSynchronize();

    const double t0 = get_time();
    getKeys<<<NBLOCK,NTHREAD>>>(numBodies, d_domain, d_bodyPos, d_key.ptr, d_value.ptr);
    sort(numBodies, d_key.ptr, d_value.ptr);
    permuteBodies<<<NBLOCK,NTHREAD>>>(numBodies, d_value, d_bodyPos, d_bodyPos2);

    cuda_mem<int> d_bodyBeginIdx, d_bodyEndIdx;
    cuda_mem<unsigned long long> d_key2;
    d_bodyBeginIdx.alloc(numBodies);
    d_bodyEndIdx.alloc(numBodies);
    d_key2.alloc(numBodies);

    unsigned long long mask = 0;
    for (int i=0; i<NBITS; i++) {
      mask <<= 3;
      if (i < levelSplit)
	mask |= 0x7;
    }
    maskKeys<<<NBLOCK,NTHREAD>>>(numBodies, mask, d_key.ptr, d_key2.ptr, d_bodyBeginIdx, d_bodyEndIdx);
    scan(numBodies, d_key.ptr, d_bodyBeginIdx.ptr);
    scan(numBodies, d_key2.ptr, d_bodyEndIdx.ptr);
    getTargetRange<<<NBLOCK,NTHREAD>>>(numBodies, d_bodyBeginIdx, d_bodyEndIdx, d_targetRange);
    kernelSuccess("groupTargets");
    const double dt = get_time() - t0;
    fprintf(stdout,"Make groups          : %.7f s\n", dt);
    int numTargets;
    CUDA_SAFE_CALL(cudaMemcpyFromSymbol(&numTargets, numTargetGlob, sizeof(int)));
    return numTargets;
  }
};
