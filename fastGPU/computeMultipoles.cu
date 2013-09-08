#include "Treecode.h"

#include "cuda_primitives.h"

namespace multipoles {

  static __device__ __forceinline__ void addMonopole(double4 &_M, const float4 ptcl) {
    const float x = ptcl.x;
    const float y = ptcl.y;
    const float z = ptcl.z;
    const float m = ptcl.w;
    float4 M = {m*x,m*y,m*z,m};
#pragma unroll
    for (int i=WARP_SIZE2-1; i>=0; i--) {
      M.x += __shfl_xor(M.x, 1<<i);
      M.y += __shfl_xor(M.y, 1<<i);
      M.z += __shfl_xor(M.z, 1<<i);
      M.w += __shfl_xor(M.w, 1<<i);
    }
    _M.x += M.x;
    _M.y += M.y;
    _M.z += M.z;
    _M.w += M.w;
  }

  static __device__ __forceinline__ void addQuadrupole(double6 &_Q, const float4 ptcl) {
    const float x = ptcl.x;
    const float y = ptcl.y;
    const float z = ptcl.z;
    const float m = ptcl.w;
    float6 Q;
    Q.xx = m * x*x;
    Q.yy = m * y*y;
    Q.zz = m * z*z;
    Q.xy = m * x*y;
    Q.xz = m * x*z;
    Q.yz = m * y*z;
#pragma unroll
    for (int i=WARP_SIZE2-1; i>=0; i--) {
      Q.xx += __shfl_xor(Q.xx, 1<<i);
      Q.yy += __shfl_xor(Q.yy, 1<<i);
      Q.zz += __shfl_xor(Q.zz, 1<<i);
      Q.xy += __shfl_xor(Q.xy, 1<<i);
      Q.xz += __shfl_xor(Q.xz, 1<<i);
      Q.yz += __shfl_xor(Q.yz, 1<<i);
    }
    _Q.xx += Q.xx;
    _Q.yy += Q.yy;
    _Q.zz += Q.zz;
    _Q.xy += Q.xy;
    _Q.xz += Q.xz;
    _Q.yz += Q.yz;
  }

  __device__ unsigned int nflops = 0;

  template<int NTHREAD2>
  static __global__ __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
  void computeCellMultipoles(
    const int nPtcl,
    const int nCells,
    const CellData *cells,
    const float4* __restrict__ ptclPos,
    const float inv_theta,
    float4 *sizeList,
    float4 *monopoleList,
    float4 *quadrpl0List,
    float2 *quadrpl1List)
  {
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);

    const int NWARP2  = NTHREAD2 - WARP_SIZE2;
    const int cellIdx = (blockIdx.x<<NWARP2) + warpIdx;
    if (cellIdx >= nCells) return;

    /* a warp compute properties of each cell */

    const CellData cell = cells[cellIdx];

    const float huge = 1e10f;
    float3 rmin = {+huge,+huge,+huge};
    float3 rmax = {-huge,-huge,-huge};
    double4 M;
    double6 Q;

    unsigned int nflop = 0;
    const int firstBody = cell.pbeg();
    const int  lastBody = cell.pend();

    for (int i = firstBody; i < lastBody; i += WARP_SIZE)
    {
      nflop++;
      float4 ptcl = ptclPos[min(i+laneIdx,lastBody-1)];
      if (i + laneIdx >= lastBody) ptcl.w = 0.0f;
      addBoxSize(rmin, rmax, make_float3(ptcl.x,ptcl.y,ptcl.z));
      addMonopole(M, ptcl);
      addQuadrupole(Q, ptcl);
    }


    if (laneIdx == 0)
    {
      const double inv_mass = 1.0/M.w;
      M.x *= inv_mass;
      M.y *= inv_mass;
      M.z *= inv_mass;
      Q.xx = Q.xx*inv_mass - M.x*M.x;
      Q.yy = Q.yy*inv_mass - M.y*M.y;
      Q.zz = Q.zz*inv_mass - M.z*M.z;
      Q.xy = Q.xy*inv_mass - M.x*M.y;
      Q.xz = Q.xz*inv_mass - M.x*M.z;
      Q.yz = Q.yz*inv_mass - M.y*M.z;

      const float3 cvec = {(rmax.x+rmin.x)*0.5f, (rmax.y+rmin.y)*0.5f, (rmax.z+rmin.z)*0.5f};
      const float3 hvec = {(rmax.x-rmin.x)*0.5f, (rmax.y-rmin.y)*0.5f, (rmax.z-rmin.z)*0.5f};
      const float3 com = {M.x, M.y, M.z};
      const float dx = cvec.x - com.x;
      const float dy = cvec.y - com.y;
      const float dz = cvec.z - com.z;
      const float  s = sqrt(dx*dx + dy*dy + dz*dz);
      const float  l = max(2.0f*max(hvec.x, max(hvec.y, hvec.z)), 1.0e-6f);
      const float cellOp = l*inv_theta + s;
      const float cellOp2 = cellOp*cellOp;

      atomicAdd(&nflops, nflop);

      sizeList[cellIdx] = (float4){com.x, com.y, com.z, cellOp2};
      monopoleList[cellIdx] = (float4){M.x, M.y, M.z, M.w};  
      quadrpl0List[cellIdx] = (float4){Q.xx, Q.yy, Q.zz, Q.xy};
      quadrpl1List[cellIdx] = (float2){Q.xz, Q.yz};
    }
  }

};

void Treecode::computeMultipoles()
{
  d_sourceCenter    .realloc(nCells);
  d_cellMonopole.realloc(nCells);
  d_cellQuad0   .realloc(nCells);
  d_cellQuad1   .realloc(nCells);

  const int NTHREAD2 = 8;
  const int NTHREAD  = 1<< NTHREAD2;
  const int NWARP    = 1<<(NTHREAD2-WARP_SIZE2);
  const int nblock   = (nCells-1)/NWARP + 1;

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&multipoles::computeCellMultipoles<NTHREAD2>,cudaFuncCachePreferL1));
  cudaDeviceSynchronize();
  const double t0 = get_time();
  multipoles::computeCellMultipoles<NTHREAD2><<<nblock,NTHREAD>>>(
      nPtcl, nCells, d_cellDataList, (float4*)d_ptclPos.ptr,
      1.0/theta,
      d_sourceCenter, d_cellMonopole, d_cellQuad0, d_cellQuad1);
  kernelSuccess("cellMultipole");
  const double dt = get_time() - t0;
  fprintf(stdout,"Upward pass          : %.7f s\n", dt);

}