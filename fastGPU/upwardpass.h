#pragma once

namespace {
  static __device__ __forceinline__
    float3 setCenter(int size, float4 * __restrict__ position, int stride) {
    float mass;
    float3 center;
    for (int i=0; i<size; i++) {
      const float4 pos = position[stride*i];
      mass += pos.w;
      center.x += pos.w * pos.x;
      center.y += pos.w * pos.y;
      center.z += pos.w * pos.z;
    }
    const float invM = 1.0f / mass;
    center.x *= invM;
    center.y *= invM;
    center.z *= invM;
    return center;
  }

  static __device__ __forceinline__
    void pairMinMax(float3 & xmin, float3 & xmax,
		    float4 reg_min, float4 reg_max) {
    xmin.x = fminf(xmin.x, reg_min.x);
    xmin.y = fminf(xmin.y, reg_min.y);
    xmin.z = fminf(xmin.z, reg_min.z);
    xmax.x = fmaxf(xmax.x, reg_max.x);
    xmax.y = fmaxf(xmax.y, reg_max.y);
    xmax.z = fmaxf(xmax.z, reg_max.z);
  }

  static __global__ __launch_bounds__(NTHREAD)
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
    void P2M(const int numLeafs,
	     const float invTheta,
	     int * leafCells,
	     CellData * cells,
	     float4 * sourceCenter,
	     float4 * bodyPos,
	     float4 * cellXmin,
	     float4 * cellXmax,
	     float4 * Multipole) {
    const int leafIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (leafIdx >= numLeafs) return;
    int cellIdx = leafCells[leafIdx];
    const CellData cell = cells[cellIdx];
    const int begin = cell.body();
    const int size = cell.nbody();
    const int end = begin + size;
    const float3 center = setCenter(size,bodyPos+begin,1);
    float4 M[3];
    const float huge = 1e10f;
    float3 Xmin = {+huge, +huge, +huge};
    float3 Xmax = {-huge, -huge, -huge};
    for( int i=begin; i<end; i++ ) {
      float4 body = bodyPos[i];
      float dx = center.x - body.x;
      float dy = center.y - body.y;
      float dz = center.z - body.z;
      M[0].w += body.w;
      M[1].x += body.w * dx * dx;
      M[1].y += body.w * dy * dy;
      M[1].z += body.w * dz * dz;
      M[1].w += body.w * dx * dy;
      M[2].x += body.w * dx * dz;
      M[2].y += body.w * dy * dz;
      pairMinMax(Xmin, Xmax, body, body);
    }
    M[0].x = center.x;
    M[0].y = center.y;
    M[0].z = center.z;
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float dx = X.x - center.x;
    const float dy = X.y - center.y;
    const float dz = X.z - center.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float cellOp = l*invTheta + s;
    const float cellOp2 = cellOp*cellOp;
    sourceCenter[cellIdx] = make_float4(center.x, center.y, center.z, cellOp2);
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = (float4){M[i].x,M[i].y,M[i].z,M[i].w};
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
  }

  static __global__ __launch_bounds__(NTHREAD)
    void M2M(const int level,
	     const float invTheta,
	     int2 * levelRange,
	     CellData * cells,
	     float4 * sourceCenter,
	     float4 * bodyPos,
	     float4 * cellXmin,
	     float4 * cellXmax,
	     float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + levelRange[level].x;
    if (cellIdx >= levelRange[level].y) return;
    const CellData cell = cells[cellIdx];
    const int begin = cell.child();
    const int size = cell.nchild();
    const int end = begin + size;
    if (cell.isLeaf()) return;
    const float3 center = setCenter(size,Multipole+3*begin,3);
    float Mi[12];
    float4 Mj[3];
    const float huge = 1e10f;
    float3 Xmin = {+huge, +huge, +huge};
    float3 Xmax = {-huge, -huge, -huge};
    for( int i=begin; i<end; i++ ) {
      Mj[0] = Multipole[3*i];
      float dx = center.x - Mj[0].x;
      float dy = center.y - Mj[0].y;
      float dz = center.z - Mj[0].z;
      Mi[3] += Mj[0].w;
      Mi[4] += Mj[1].x;
      Mi[5] += Mj[1].y;
      Mi[6] += Mj[1].z;
      Mi[7] += Mj[1].w;
      Mi[8] += Mj[2].x;
      Mi[9] += Mj[2].y;
      Mi[4] += Mj[0].w * dx * dx;
      Mi[5] += Mj[0].w * dy * dy;
      Mi[6] += Mj[0].w * dz * dz;
      Mi[7] += Mj[0].w * dx * dy;
      Mi[8] += Mj[0].w * dx * dz;
      Mi[9] += Mj[0].w * dy * dz;
      pairMinMax(Xmin, Xmax, cellXmin[i], cellXmax[i]);
    }
    Mi[0] = center.x;
    Mi[1] = center.y;
    Mi[2] = center.z;
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float dx = X.x - center.x;
    const float dy = X.y - center.y;
    const float dz = X.z - center.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float cellOp = l*invTheta + s;
    const float cellOp2 = cellOp*cellOp;
    sourceCenter[cellIdx] = make_float4(center.x, center.y, center.z, cellOp2);
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = make_float4(Mi[4*i+0],Mi[4*i+1],Mi[4*i+2],Mi[4*i+3]);
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
  }

  static __global__ __launch_bounds__(NTHREAD)
    void normalize(const int numCells, float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const float invM = 1.0 / Multipole[3*cellIdx].w;
    for (int i=1; i<3; i++) {
      Multipole[3*cellIdx+i].x *= invM;
      Multipole[3*cellIdx+i].y *= invM;
      Multipole[3*cellIdx+i].z *= invM;
      Multipole[3*cellIdx+i].w *= invM;
    }
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
			      bodyPos.d(),cellXmin.d(),cellXmax.d(),Multipole.d());
      kernelSuccess("M2M");
    }
    numCells = sourceCells.size();
    NBLOCK = (numCells - 1) / NTHREAD + 1;
    normalize<<<NBLOCK,NTHREAD>>>(numCells, Multipole.d()); 
    const double dt = get_time() - t0;
    fprintf(stdout,"Upward pass          : %.7f s\n", dt);
  }
};
