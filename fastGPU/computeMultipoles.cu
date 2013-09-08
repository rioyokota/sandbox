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
      M.x += shfl_xor(M.x, 1<<i);
      M.y += shfl_xor(M.y, 1<<i);
      M.z += shfl_xor(M.z, 1<<i);
      M.w += shfl_xor(M.w, 1<<i);
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
      Q.xx += shfl_xor(Q.xx, 1<<i);
      Q.yy += shfl_xor(Q.yy, 1<<i);
      Q.zz += shfl_xor(Q.zz, 1<<i);
      Q.xy += shfl_xor(Q.xy, 1<<i);
      Q.xz += shfl_xor(Q.xz, 1<<i);
      Q.yz += shfl_xor(Q.yz, 1<<i);
    }
    _Q.xx += Q.xx;
    _Q.yy += Q.yy;
    _Q.zz += Q.zz;
    _Q.xy += Q.xy;
    _Q.xz += Q.xz;
    _Q.yz += Q.yz;
  }

  __device__ unsigned int nflops = 0;

  template<int NTHREAD2, typename real_t>
  static __global__ __launch_bounds__(1<<NTHREAD2, 1024/(1<<NTHREAD2))
  void computeCellMultipoles(
    const int nPtcl,
    const int nCells,
    const CellData *cells,
    const float4* __restrict__ ptclPos,
    const real_t inv_theta,
    typename vec<4,real_t>::type *sizeList,
    typename vec<4,real_t>::type *monopoleList,
    typename vec<4,real_t>::type *quadrpl0List,
    typename vec<2,real_t>::type *quadrpl1List)
  {
    const int warpIdx = threadIdx.x >> WARP_SIZE2;
    const int laneIdx = threadIdx.x & (WARP_SIZE-1);

    const int NWARP2  = NTHREAD2 - WARP_SIZE2;
    const int cellIdx = (blockIdx.x<<NWARP2) + warpIdx;
    if (cellIdx >= nCells) return;

    /* a warp compute properties of each cell */

    const CellData cell = cells[cellIdx];

    const real_t huge = static_cast<real_t>(1e10f);
    typename vec<3,real_t>::type rmin = {+huge,+huge,+huge};
    typename vec<3,real_t>::type rmax = {-huge,-huge,-huge};
    double4 M;
    double6 Q;

    unsigned int nflop = 0;
#if 0
    if (cell.isNode())
    {
    }
    else
#endif
    {
      const int firstBody = cell.pbeg();
      const int  lastBody = cell.pend();

      for (int i = firstBody; i < lastBody; i += WARP_SIZE)
      {
	nflop++;
	float4 ptcl = ptclPos[min(i+laneIdx,lastBody-1)];
	if (i + laneIdx >= lastBody) ptcl.w = 0.0f;
	addBoxSize(rmin, rmax, Position<real_t>(ptcl.x,ptcl.y,ptcl.z));
	addMonopole(M, ptcl);
	addQuadrupole(Q, ptcl);
      }
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

      const Position<real_t> cvec((rmax.x+rmin.x)*real_t(0.5f), (rmax.y+rmin.y)*real_t(0.5f), (rmax.z+rmin.z)*real_t(0.5f));
      const Position<real_t> hvec((rmax.x-rmin.x)*real_t(0.5f), (rmax.y-rmin.y)*real_t(0.5f), (rmax.z-rmin.z)*real_t(0.5f));
      const Position<real_t> com(M.x, M.y, M.z);
      const real_t dx = cvec.x - com.x;
      const real_t dy = cvec.y - com.y;
      const real_t dz = cvec.z - com.z;
      const real_t  s = sqrt(dx*dx + dy*dy + dz*dz);
      const real_t  l = max(static_cast<real_t>(2.0f)*max(hvec.x, max(hvec.y, hvec.z)), static_cast<real_t>(1.0e-6f));
      const real_t cellOp = l*inv_theta + s;
      const real_t cellOp2 = cellOp*cellOp;

      atomicAdd(&nflops, nflop);

      sizeList[cellIdx] = (float4){com.x, com.y, com.z, cellOp2};
      monopoleList[cellIdx] = (float4){M.x, M.y, M.z, M.w};  
      quadrpl0List[cellIdx] = (float4){Q.xx, Q.yy, Q.zz, Q.xy};
      quadrpl1List[cellIdx] = (float2){Q.xz, Q.yz};
    }
  }

};

  template<typename real_t>
void Treecode<real_t>::computeMultipoles()
{
  d_sourceCenter    .realloc(nCells);
  d_cellMonopole.realloc(nCells);
  d_cellQuad0   .realloc(nCells);
  d_cellQuad1   .realloc(nCells);

  const int NTHREAD2 = 8;
  const int NTHREAD  = 1<< NTHREAD2;
  const int NWARP    = 1<<(NTHREAD2-WARP_SIZE2);
  const int nblock   = (nCells-1)/NWARP + 1;

  CUDA_SAFE_CALL(cudaFuncSetCacheConfig(&multipoles::computeCellMultipoles<NTHREAD2,real_t>,cudaFuncCachePreferL1));
  cudaDeviceSynchronize();
  const double t0 = rtc();
  multipoles::computeCellMultipoles<NTHREAD2,real_t><<<nblock,NTHREAD>>>(
      nPtcl, nCells, d_cellDataList, (float4*)d_ptclPos.ptr,
      1.0/theta,
      d_sourceCenter, d_cellMonopole, d_cellQuad0, d_cellQuad1);
  kernelSuccess("cellMultipole");
  const double dt = rtc() - t0;
  fprintf(stdout,"Upward pass          : %.7f s\n", dt);

}

#include "TreecodeInstances.h"

