#pragma once

namespace {
  static __device__ __forceinline__
    fvec4 setCenter(const int begin, const int end, fvec4 * posGlob) {
    fvec4 center;
    for (int i=begin; i<end; i++) {
      const fvec4 pos = posGlob[i];
      center[0] += pos[3] * pos[0];
      center[1] += pos[3] * pos[1];
      center[2] += pos[3] * pos[2];
      center[3] += pos[3];
    }
    const float invM = 1.0f / center[3];
    center[0] *= invM;
    center[1] *= invM;
    center[2] *= invM;
    return center;
  }

  static __device__ __forceinline__
    void getMinMax(const int begin, const int end,
		   fvec4 * XminGlob, fvec4 * XmaxGlob,
		   fvec4 & Xmin, fvec4 & Xmax) {
    for (int i=begin; i<end; i++) {
      Xmin = min(Xmin, XminGlob[i]);
      Xmax = max(Xmax, XmaxGlob[i]);
    }
  }

  static __device__ __forceinline__
    void P2M(const int begin,
	     const int end,
	     fvec4 * bodyPos,
	     const fvec4 center,
	     float * M) {
    for (int i=begin; i<end; i++) {
      fvec4 body = bodyPos[i];
      float dx = center[0] - body[0];
      float dy = center[1] - body[1];
      float dz = center[2] - body[2];
      M[0] += body[3];
      M[1] += body[3] * dx;
      M[2] += body[3] * dy;
      M[3] += body[3] * dz;
      M[4] += .5 * body[3] * dx * dx;
      M[5] += .5 * body[3] * dy * dy;
      M[6] += .5 * body[3] * dz * dz;
      M[7] += body[3] * dx * dy;
      M[8] += body[3] * dx * dz;
      M[9] += body[3] * dy * dz;
    }
  }

  static __device__ __forceinline__
    void M2M(const int begin,
	     const int end,
	     const fvec4 Xi,
	     fvec4 * sourceCenter,
	     float4 * Multipole,
	     float * Mi) {
    for (int i=begin; i<end; i++) {
      float * Mj = (float*) &Multipole[3*i];
      fvec4 Xj = sourceCenter[i];
      float dx = Xi[0] - Xj[0];
      float dy = Xi[1] - Xj[1];
      float dz = Xi[2] - Xj[2];
      for (int j=0; j<10; j++) Mi[j] += Mj[j];
      Mi[4] += .5 * Mj[0] * dx * dx;
      Mi[5] += .5 * Mj[0] * dy * dy;
      Mi[6] += .5 * Mj[0] * dz * dz;
      Mi[7] += Mj[0] * dx * dy;
      Mi[8] += Mj[0] * dx * dz;
      Mi[9] += Mj[0] * dy * dz;
    }
  }

  static __global__ __launch_bounds__(NTHREAD)
    void upwardPass(const int level,
		    int2 * levelRange,
		    CellData * cells,
		    float4 * sourceCenter,
		    float4 * bodyPos,
		    fvec4 * cellXmin,
		    fvec4 * cellXmax,
		    float4 * Multipole) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x + levelRange[level].x;
    if (cellIdx >= levelRange[level].y) return;
    const CellData cell = cells[cellIdx];
    fvec4 Xmin = 1e10f;
    fvec4 Xmax = -1e10f;
    fvec4 center;
    float M[12];
    if (cell.isLeaf()) {
      const int begin = cell.body();
      const int end = begin + cell.nbody();
      center = setCenter(begin, end, reinterpret_cast<fvec4*>(bodyPos));
      getMinMax(begin, end, reinterpret_cast<fvec4*>(bodyPos), reinterpret_cast<fvec4*>(bodyPos), Xmin, Xmax);
      P2M(begin, end, reinterpret_cast<fvec4*>(bodyPos), center, M);
    } else {
      const int begin = cell.child();
      const int end = begin + cell.nchild();
      center = setCenter(begin, end, reinterpret_cast<fvec4*>(sourceCenter));
      getMinMax(begin, end, cellXmin, cellXmax, Xmin, Xmax);
      M2M(begin, end, center, reinterpret_cast<fvec4*>(sourceCenter), Multipole, M); 
    }
    sourceCenter[cellIdx] = make_float4(center[0],center[1],center[2],center[3]);
    cellXmin[cellIdx] = Xmin;
    cellXmax[cellIdx] = Xmax;
    for (int i=0; i<3; i++) Multipole[3*cellIdx+i] = (float4){M[4*i+0],M[4*i+1],M[4*i+2],M[4*i+3]};
  }

  static __global__ __launch_bounds__(NTHREAD)
    void setMAC(const int numCells, const float invTheta, float4 * sourceCenter,
		fvec4 * cellXmin, fvec4 * cellXmax) {
    const int cellIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (cellIdx >= numCells) return;
    const fvec4 Xmin = cellXmin[cellIdx];
    const fvec4 Xmax = cellXmax[cellIdx];
    const float4 Xi = sourceCenter[cellIdx];
    const float3 X = {(Xmax[0]+Xmin[0])*0.5f, (Xmax[1]+Xmin[1])*0.5f, (Xmax[2]+Xmin[2])*0.5f};
    const float3 R = {(Xmax[0]-Xmin[0])*0.5f, (Xmax[1]-Xmin[1])*0.5f, (Xmax[2]-Xmin[2])*0.5f};
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
    const int numCells = sourceCells.size();
    const double t0 = get_time();
    cudaVec<fvec4> cellXmin(numCells);
    cudaVec<fvec4> cellXmax(numCells);
    levelRange.d2h();
    for (int level=numLevels; level>=1; level--) {
      const int numCellsPerLevel = levelRange[level].y - levelRange[level].x;
      const int NBLOCK = (numCellsPerLevel - 1) / NTHREAD + 1;
      upwardPass<<<NBLOCK,NTHREAD>>>(level,levelRange.d(),sourceCells.d(),sourceCenter.d(),
				     bodyPos.d(),cellXmin.d(),cellXmax.d(),Multipole.d());
      kernelSuccess("upwardPass");
    }
    int NBLOCK = (numCells - 1) / NTHREAD + 1;
    setMAC<<<NBLOCK,NTHREAD>>>(numCells, 1.0/theta, sourceCenter.d(),
			       cellXmin.d(),cellXmax.d());
    normalize<<<NBLOCK,NTHREAD>>>(numCells, Multipole.d());
    const double dt = get_time() - t0;
    fprintf(stdout,"Upward pass          : %.7f s\n", dt);
  }
};
