#pragma once

namespace {
  static __device__ __forceinline__
    float4 setCenter(const int begin, const int end, float4 * posGlob) {
    float4 center;
    for (int i=begin; i<end; i++) {
      const float4 pos = posGlob[i];
      center.x += pos.w * pos.x;
      center.y += pos.w * pos.y;
      center.z += pos.w * pos.z;
      center.w += pos.w;
    }
    const float invM = 1.0f / center.w;
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
    void P2M2(const int numCells,
	     CellData * cells,
	     float4 * sourceCenter,
	     float4 * bodyPos,
	     float4 * cellXmin,
	     float4 * cellXmax,
	     float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const CellData cell = cells[cellIdx];
    if (cell.isNode()) return;
    const int begin = cell.body();
    const int end = begin + cell.nbody();
    const float4 center = setCenter(begin, end, bodyPos);
    float M[12];
    const float huge = 1e10f;
    float3 Xmin = {+huge, +huge, +huge};
    float3 Xmax = {-huge, -huge, -huge};
    for (int i=begin; i<end; i++) {
      float4 body = bodyPos[i];
      float dx = center.x - body.x;
      float dy = center.y - body.y;
      float dz = center.z - body.z;
      M[0] += body.w;
      M[1] += body.w * dx;
      M[2] += body.w * dy;
      M[3] += body.w * dz;
      M[4] += .5 * body.w * dx * dx;
      M[5] += .5 * body.w * dy * dy;
      M[6] += .5 * body.w * dz * dz;
      M[7] += body.w * dx * dy;
      M[8] += body.w * dx * dz;
      M[9] += body.w * dy * dz;
      pairMinMax(Xmin, Xmax, body, body);
    }
    sourceCenter[cellIdx] = center;
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = (float4){M[4*i+0],M[4*i+1],M[4*i+2],M[4*i+3]};
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
  }

  static __global__ __launch_bounds__(NTHREAD)
    void M2M2(const int level,
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
    if (cell.isLeaf()) return;
    const int begin = cell.child();
    const int end = begin + cell.nchild();
    const float4 Xi = setCenter(begin,end,sourceCenter);
    float Mi[12];
    const float huge = 1e10f;
    float3 Xmin = {+huge, +huge, +huge};
    float3 Xmax = {-huge, -huge, -huge};
    for (int i=begin; i<end; i++) {
      float * Mj = (float*) &Multipole[3*i];
      float4 Xj = sourceCenter[i];
      float dx = Xi.x - Xj.x;
      float dy = Xi.y - Xj.y;
      float dz = Xi.z - Xj.z;
      for (int j=0; j<10; j++) Mi[j] += Mj[j];
      Mi[4] += .5 * Mj[0] * dx * dx;
      Mi[5] += .5 * Mj[0] * dy * dy;
      Mi[6] += .5 * Mj[0] * dz * dz;
      Mi[7] += Mj[0] * dx * dy;
      Mi[8] += Mj[0] * dx * dz;
      Mi[9] += Mj[0] * dy * dz;
      pairMinMax(Xmin, Xmax, cellXmin[i], cellXmax[i]);
    }
    sourceCenter[cellIdx] = Xi;
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = make_float4(Mi[4*i+0],Mi[4*i+1],Mi[4*i+2],Mi[4*i+3]);
    cellXmin[cellIdx] = make_float4(Xmin.x, Xmin.y, Xmin.z, 0.0f);
    cellXmax[cellIdx] = make_float4(Xmax.x, Xmax.y, Xmax.z, 0.0f);
  }

  static __global__ __launch_bounds__(NTHREAD)
    void P2M(const int numCells,
	     CellData * cells,
	     float4 * sourceCenter,
	     float4 * bodyPos,
	     float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const CellData cell = cells[cellIdx];
    if (cell.isNode()) return;
    const int begin = cell.body();
    const int end = begin + cell.nbody();
    const float4 center = sourceCenter[cellIdx];
    float M[12];
    for (int i=begin; i<end; i++) {
      float4 body = bodyPos[i];
      float dx = center.x - body.x;
      float dy = center.y - body.y;
      float dz = center.z - body.z;
      M[0] += body.w;
      M[1] += body.w * dx;
      M[2] += body.w * dy;
      M[3] += body.w * dz;
      M[4] += .5 * body.w * dx * dx;
      M[5] += .5 * body.w * dy * dy;
      M[6] += .5 * body.w * dz * dz;
      M[7] += body.w * dx * dy;
      M[8] += body.w * dx * dz;
      M[9] += body.w * dy * dz;
    }
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = (float4){M[4*i+0],M[4*i+1],M[4*i+2],M[4*i+3]};
  }

  static __global__ __launch_bounds__(NTHREAD)
    void M2M(const int level,
	     int2 * levelRange,
	     CellData * cells,
	     float4 * sourceCenter,
	     float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + levelRange[level].x;
    if (cellIdx >= levelRange[level].y) return;
    const CellData cell = cells[cellIdx];
    if (cell.isLeaf()) return;
    const int begin = cell.child();
    const int end = begin + cell.nchild();
    const float4 Xi = sourceCenter[cellIdx];
    float Mi[12];
    for (int i=begin; i<end; i++) {
      float * Mj = (float*) &Multipole[3*i];
      float4 Xj = sourceCenter[i];
      float dx = Xi.x - Xj.x;
      float dy = Xi.y - Xj.y;
      float dz = Xi.z - Xj.z;
      for (int j=0; j<10; j++) Mi[j] += Mj[j];
      Mi[4] += .5 * Mj[0] * dx * dx;
      Mi[5] += .5 * Mj[0] * dy * dy;
      Mi[6] += .5 * Mj[0] * dz * dz;
      Mi[7] += Mj[0] * dx * dy;
      Mi[8] += Mj[0] * dx * dz;
      Mi[9] += Mj[0] * dy * dz;
    }
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = make_float4(Mi[4*i+0],Mi[4*i+1],Mi[4*i+2],Mi[4*i+3]);
  }

  static __global__ __launch_bounds__(NTHREAD)
    void setMAC(const int numCells, const float invTheta, float4 * sourceCenter,
		float4 * cellXmin, float4 * cellXmax) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const float4 Xmin = cellXmin[cellIdx];
    const float4 Xmax = cellXmax[cellIdx];
    const float4 Xi = sourceCenter[cellIdx];
    const float3 X = {(Xmax.x+Xmin.x)*0.5f, (Xmax.y+Xmin.y)*0.5f, (Xmax.z+Xmin.z)*0.5f};
    const float3 R = {(Xmax.x-Xmin.x)*0.5f, (Xmax.y-Xmin.y)*0.5f, (Xmax.z-Xmin.z)*0.5f};
    const float dx = X.x - Xi.x;
    const float dy = X.y - Xi.y;
    const float dz = X.z - Xi.z;
    const float  s = sqrt(dx*dx + dy*dy + dz*dz);
    const float  l = max(2.0f*max(R.x, max(R.y, R.z)), 1.0e-6f);
    const float MAC = l * invTheta + s;
    const float MAC2 = MAC * MAC;
    sourceCenter[cellIdx].w = MAC2;
  }

  static __global__ __launch_bounds__(NTHREAD)
    void normalize(const int numCells, float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const float invM = 1.0 / Multipole[3*cellIdx].x;
    Multipole[3*cellIdx].y *= invM;
    Multipole[3*cellIdx].z *= invM;
    Multipole[3*cellIdx].w *= invM;
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
    const double t0 = get_time();
    cudaVec<float4> cellXmin(numCells);
    cudaVec<float4> cellXmax(numCells);
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&P2M,cudaFuncCachePreferL1));
    //CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&M2M,cudaFuncCachePreferL1));
    //cudaDeviceSynchronize();
    P2M2<<<NBLOCK,NTHREAD>>>(numCells,sourceCells.d(),sourceCenter.d(),
			     bodyPos.d(),cellXmin.d(),cellXmax.d(),Multipole.d());
    kernelSuccess("P2M2");
    levelRange.d2h();
    for (int level=numLevels; level>=1; level--) {
      numCells = levelRange[level].y - levelRange[level].x;
      NBLOCK = (numCells - 1) / NTHREAD + 1;
      M2M2<<<NBLOCK,NTHREAD>>>(level,levelRange.d(),sourceCells.d(),sourceCenter.d(),
			       bodyPos.d(),cellXmin.d(),cellXmax.d(),Multipole.d());
      kernelSuccess("M2M2");
    }
    Multipole.zeros();
    numCells = sourceCells.size();
    NBLOCK = (numCells-1) / NTHREAD + 1;
    P2M<<<NBLOCK,NTHREAD>>>(numCells,sourceCells.d(),sourceCenter.d(),
			    bodyPos.d(),Multipole.d());
    kernelSuccess("P2M");
    for (int level=numLevels; level>=1; level--) {
      numCells = levelRange[level].y - levelRange[level].x;
      NBLOCK = (numCells - 1) / NTHREAD + 1;
      M2M<<<NBLOCK,NTHREAD>>>(level,levelRange.d(),sourceCells.d(),
			      sourceCenter.d(),Multipole.d());
      kernelSuccess("M2M");
    }
    numCells = sourceCells.size();
    NBLOCK = (numCells - 1) / NTHREAD + 1;
    setMAC<<<NBLOCK,NTHREAD>>>(numCells, 1.0/theta, sourceCenter.d(),
			       cellXmin.d(),cellXmax.d());
    normalize<<<NBLOCK,NTHREAD>>>(numCells, Multipole.d());
    const double dt = get_time() - t0;
    fprintf(stdout,"Upward pass          : %.7f s\n", dt);
  }
};
