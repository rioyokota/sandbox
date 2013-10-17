#pragma once

namespace {
  static __device__ __forceinline__
    void pairMinMax(float3 &xmin, float3 &xmax,
		    float4 reg_min, float4 reg_max) {
    xmin.x = min(xmin.x, reg_min.x);
    xmin.y = min(xmin.y, reg_min.y);
    xmin.z = min(xmin.z, reg_min.z);
    xmax.x = max(xmax.x, reg_max.x);
    xmax.y = max(xmax.y, reg_max.y);
    xmax.z = max(xmax.z, reg_max.z);
  }

  static __device__ __forceinline__
    void addMultipole(double4 * __restrict__ _M, const float4 body) {
    const float x = body.x;
    const float y = body.y;
    const float z = body.z;
    const float m = body.w;
    float4 M[3];
    M[0].x = m*x;
    M[0].y = m*y;
    M[0].z = m*z;
    M[0].w = m;
    M[1].x = m*x*x;
    M[1].y = m*y*y;
    M[1].z = m*z*z;
    M[1].w = m*x*y;
    M[2].x = m*x*z;
    M[2].y = m*y*z;
    M[2].z = 0.0f;
    M[2].w = 0.0f;
#pragma unroll
    for (int j=0; j<3; j++) {
#pragma unroll
      for (int i=WARP_SIZE2-1; i>=0; i--) {
	M[j].x += __shfl_xor(M[j].x, 1<<i);
	M[j].y += __shfl_xor(M[j].y, 1<<i);
	M[j].z += __shfl_xor(M[j].z, 1<<i);
	M[j].w += __shfl_xor(M[j].w, 1<<i);
      }
      _M[j].x += M[j].x;
      _M[j].y += M[j].y;
      _M[j].z += M[j].z;
      _M[j].w += M[j].w;
    }
  }

  static __global__
    void collectLeafs(const int numCells,
                      const CellData * sourceCells,
                      int * leafCells) {
    const int NWARP = 1 << (NTHREAD2 - WARP_SIZE2);
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int cellIdx = blockDim.x * blockIdx.x + threadIdx.x;
    const CellData cell = sourceCells[min(cellIdx, numCells-1)];
    const bool isLeaf = cellIdx < numCells & cell.isLeaf();
    const int numLeafsLane = exclusiveScanBool(isLeaf);
    const int numLeafsWarp = reduceBool(isLeaf);
    __shared__ int numLeafsBase[NWARP];
    int & numLeafsScan = numLeafsBase[warpIdx];
    if (laneIdx == 0 && numLeafsWarp > 0)
      numLeafsScan = atomicAdd(&numLeafsGlob, numLeafsWarp);
    if (isLeaf)
      leafCells[numLeafsScan+numLeafsLane] = cellIdx;
  }

  static __global__ __launch_bounds__(NTHREAD)
    void getMultipoles(const int numBodies,
		       const int numCells,
		       const CellData * __restrict__ cells,
		       const float4 * __restrict__ bodyPos,
		       const float invTheta,
		       float4 * sourceCenter,
		       float4 * Multipole) {
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int NWARP2  = NTHREAD2 - WARP_SIZE2;
    const int cellIdx = (blockIdx.x<<NWARP2) + warpIdx;
    if (cellIdx >= numCells) return;

    const CellData cell = cells[cellIdx];
    const float huge = 1e10f;
    float3 Xmin = {+huge,+huge,+huge};
    float3 Xmax = {-huge,-huge,-huge};
    double4 M[3] = {0.0};
    const int bodyBegin = cell.body();
    const int bodyEnd = cell.body() + cell.nbody();
    for (int i=bodyBegin; i<bodyEnd; i+=WARP_SIZE) {
      float4 body = bodyPos[min(i+laneIdx,bodyEnd-1)];
      if (i + laneIdx >= bodyEnd) body.w = 0.0f;
      getMinMax(Xmin, Xmax, make_float3(body.x,body.y,body.z));
      addMultipole(M, body);
    }
    const double invM = 1.0 / M[0].w;
    M[0].x *= invM;
    M[0].y *= invM;
    M[0].z *= invM;
    M[1].x = M[1].x * invM - M[0].x * M[0].x;
    M[1].y = M[1].y * invM - M[0].y * M[0].y;
    M[1].z = M[1].z * invM - M[0].z * M[0].z;
    M[1].w = M[1].w * invM - M[0].x * M[0].y;
    M[2].x = M[2].x * invM - M[0].x * M[0].z;
    M[2].y = M[2].y * invM - M[0].y * M[0].z;
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float3 com = {M[0].x, M[0].y, M[0].z};
    const float dx = X.x - com.x;
    const float dy = X.y - com.y;
    const float dz = X.z - com.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float cellOp = l*invTheta + s;
    const float cellOp2 = cellOp*cellOp;
    if (laneIdx == 0) {
      sourceCenter[cellIdx] = (float4){com.x, com.y, com.z, cellOp2};
      for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = (float4){M[i].x, M[i].y, M[i].z, M[i].w};
    }
  }

  static __global__ void P2M(const int numLeafs,
			     const float invTheta,
			     int * leafCells,
			     CellData * cells,
			     float4 * sourceCenter,
			     float4 * bodyPos,
			     float4 * cellXmin,
			     float4 * cellXmax,
			     float4  * Multipole) {
    const uint leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeafs) return;
    int cellIdx = leafCells[leafIdx];
    const CellData cell = cells[cellIdx];
    const uint begin = cell.body();
    const uint end = begin + cell.nbody();
    float4 mon = {0.0f, 0.0f, 0.0f, 0.0f};
    const float huge = 1e10f;
    float3 Xmin = {+huge,+huge,+huge};
    float3 Xmax = {-huge,-huge,-huge};
    for( int i=begin; i<end; i++ ) {
      float4 pos = bodyPos[i];
      mon.w += pos.w;
      mon.x += pos.w * pos.x;
      mon.y += pos.w * pos.y;
      mon.z += pos.w * pos.z;
      pairMinMax(Xmin, Xmax, pos, pos);
    }
    float im = 1.0/mon.w;
    if(mon.w == 0) im = 0;
    mon.x *= im;
    mon.y *= im;
    mon.z *= im;
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float3 com = {mon.x, mon.y, mon.z};
    const float dx = X.x - com.x;
    const float dy = X.y - com.y;
    const float dz = X.z - com.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float cellOp = l*invTheta + s;
    const float cellOp2 = cellOp*cellOp;
    sourceCenter[cellIdx] = (float4){com.x, com.y, com.z, cellOp2};
    Multipole[3*cellIdx+0] = make_float4(mon.x, mon.y, mon.z, mon.w);
    Multipole[3*cellIdx+1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    Multipole[3*cellIdx+2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
    return;
  }

  static __global__ void M2M(const int level,
			     const float invTheta,
			     int2 * levelRange,
			     CellData * cells,
			     float4 * sourceCenter,
			     float4 * cellXmin,
			     float4 * cellXmax,
			     float4 * Multipole) {
    const uint cellIdx = blockIdx.x * blockDim.x + threadIdx.x + levelRange[level].x;
    if (cellIdx >= levelRange[level].y) return;
    const CellData cell = cells[cellIdx];
    const uint begin = cell.child();
    const uint end = begin + cell.nchild();
    if (cell.isLeaf()) return;
    float4 mon = {0.0f, 0.0f, 0.0f, 0.0f};
    const float huge = 1e10f;
    float3 Xmin = {+huge,+huge,+huge};
    float3 Xmax = {-huge,-huge,-huge};
    for( int i=begin; i<end; i++ ) {
      float4 pos = Multipole[3*i];
      mon.w += pos.w;
      mon.x += pos.w * pos.x;
      mon.y += pos.w * pos.y;
      mon.z += pos.w * pos.z;
      pairMinMax(Xmin, Xmax, cellXmin[i], cellXmax[i]);
    }
    float im = 1.0 / mon.w;
    if(mon.w == 0) im = 0;
    mon.x *= im;
    mon.y *= im;
    mon.z *= im;
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float3 com = {mon.x, mon.y, mon.z};
    const float dx = X.x - com.x;
    const float dy = X.y - com.y;
    const float dz = X.z - com.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float cellOp = l*invTheta + s;
    const float cellOp2 = cellOp*cellOp;
    sourceCenter[cellIdx] = make_float4(com.x, com.y, com.z, cellOp2);
    Multipole[3*cellIdx+0] = make_float4(mon.x, mon.y, mon.z, mon.w);
    Multipole[3*cellIdx+1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    Multipole[3*cellIdx+2] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
    return;
  }
}

class Pass {
 public:
  void upward(const int numLeafs,
	      const int numLevels,
	      const float theta,
	      cudaVec<int2> & levelRange,
	      cudaVec<float4> & bodyPos,
	      cudaVec<CellData> & sourceCells,
	      cudaVec<float4> & sourceCenter,
	      cudaVec<float4> & Multipole) {
    const int numBodies = bodyPos.size();
    int numCells = sourceCells.size();
    int NBLOCK = (numCells-1) / NTHREAD + 1;
    cudaVec<int> leafCells(numLeafs);
    collectLeafs<<<NBLOCK,NTHREAD>>>(numCells, sourceCells.d(), leafCells.d());
    kernelSuccess("collectLeafs");
    const double t0 = get_time();
    cudaVec<float4> cellXmin(numCells);
    cudaVec<float4> cellXmax(numCells);
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&P2M,cudaFuncCachePreferL1));
    CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&M2M,cudaFuncCachePreferL1));
    cudaDeviceSynchronize();
    NBLOCK = (numLeafs - 1) / NTHREAD + 1;
    P2M<<<NBLOCK,NTHREAD>>>(numLeafs,1.0/theta,leafCells.d(),sourceCells.d(),sourceCenter.d(),
			    bodyPos.d(),cellXmin.d(),cellXmax.d(),Multipole.d());
    kernelSuccess("P2M");
    levelRange.d2h();
    for( int level=numLevels; level>=1; level-- ) {
      numCells = levelRange[level].y - levelRange[level].x;
      NBLOCK = (numCells - 1) / NTHREAD + 1;
      M2M<<<NBLOCK,NTHREAD>>>(level,1.0/theta,levelRange.d(),sourceCells.d(),sourceCenter.d(),
			      cellXmin.d(),cellXmax.d(),Multipole.d());
      kernelSuccess("M2M");
    }
    const double dt = get_time() - t0;
    fprintf(stdout,"Upward pass          : %.7f s\n", dt);
  }
};
