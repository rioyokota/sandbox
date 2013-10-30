#pragma once

namespace {
  template<int nx, int ny, int nz>
  struct Index {
    static const int                I = Index<nx,ny+1,nz-1>::I + 1;
    static const unsigned long long F = Index<nx,ny,nz-1>::F * nz;
  };

  template<int nx, int ny>
  struct Index<nx,ny,0> {
    static const int                I = Index<nx+1,0,ny-1>::I + 1;
    static const unsigned long long F = Index<nx,ny-1,0>::F * ny;
  };

  template<int nx>
  struct Index<nx,0,0> {
    static const int                I = Index<0,0,nx-1>::I + 1;
    static const unsigned long long F = Index<nx-1,0,0>::F * nx;
  };

  template<>
  struct Index<0,0,0> {
    static const int                I = 0;
    static const unsigned long long F = 1;
  };

  template<int nx, int ny, int nz, int kx=nx, int ky=ny, int kz=nz>
  struct MultipoleSum {
    static __host__ __device__ __forceinline__
    float kernel(const fvecP &C, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx,ky,kz-1>::kernel(C,M)
	+ C[Index<nx-kx,ny-ky,nz-kz>::I]*M[Index<kx,ky,kz>::I];
    }
  };

  template<int nx, int ny, int nz, int kx, int ky>
  struct MultipoleSum<nx,ny,nz,kx,ky,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvecP &C, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx,ky-1,nz>::kernel(C,M)
	+ C[Index<nx-kx,ny-ky,nz>::I]*M[Index<kx,ky,0>::I];
    }
  };

  template<int nx, int ny, int nz, int kx>
  struct MultipoleSum<nx,ny,nz,kx,0,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvecP &C, const fvecP &M) {
      return MultipoleSum<nx,ny,nz,kx-1,ny,nz>::kernel(C,M)
	+ C[Index<nx-kx,ny,nz>::I]*M[Index<kx,0,0>::I];
    }
  };

  template<int nx, int ny, int nz>
  struct MultipoleSum<nx,ny,nz,0,0,0> {
    static __host__ __device__ __forceinline__
    float kernel(const fvecP&, const fvecP&) { return 0; }
  };

  template<int nx, int ny, int nz>
  struct Kernels {
    static __host__ __device__ __forceinline__
    void power(fvecP &C, const fvec3 &dX) {
      Kernels<nx,ny+1,nz-1>::power(C,dX);
      C[Index<nx,ny,nz>::I] = C[Index<nx,ny,nz-1>::I] * dX[2] / nz;
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvecP &C, const fvecP &MJ) {
      Kernels<nx,ny+1,nz-1>::M2M(MI,C,MJ);
      MI[Index<nx,ny,nz>::I] += MultipoleSum<nx,ny,nz>::kernel(C,MJ);
    }
  };

  template<int nx, int ny>
  struct Kernels<nx,ny,0> {
    static __host__ __device__ __forceinline__
    void power(fvecP &C, const fvec3 &dX) {
      Kernels<nx+1,0,ny-1>::power(C,dX);
      C[Index<nx,ny,0>::I] = C[Index<nx,ny-1,0>::I] * dX[1] / ny;
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvecP &C, const fvecP &MJ) {
      Kernels<nx+1,0,ny-1>::M2M(MI,C,MJ);
      MI[Index<nx,ny,0>::I] += MultipoleSum<nx,ny,0>::kernel(C,MJ);
    }
  };

  template<int nx>
  struct Kernels<nx,0,0> {
    static __host__ __device__ __forceinline__
    void power(fvecP &C, const fvec3 &dX) {
      Kernels<0,0,nx-1>::power(C,dX);
      C[Index<nx,0,0>::I] = C[Index<nx-1,0,0>::I] * dX[0] / nx;
    }
    static __host__ __device__ __forceinline__
    void M2M(fvecP &MI, const fvecP &C, const fvecP &MJ) {
      Kernels<0,0,nx-1>::M2M(MI,C,MJ);
      MI[Index<nx,0,0>::I] += MultipoleSum<nx,0,0>::kernel(C,MJ);
    }
  };

  template<>
  struct Kernels<0,0,0> {
    static __host__ __device__ __forceinline__
    void power(fvecP&, const fvec3&) {}
    static __host__ __device__ __forceinline__
    void M2M(fvecP&, const fvecP&, const fvecP&) {}
  };

  __device__ __forceinline__
  void P2M(const int begin,
	   const int end,
	   const fvec4 center,
	   fvecP & Mi) {
    for (int i=begin; i<end; i++) {
      fvec4 body = tex1Dfetch(texBody,i);
      fvec3 dX = make_fvec3(center - body);
      fvecP M;
      M[0] = body[3];
      Kernels<0,0,P-1>::power(M,dX);
      Mi += M;
    }
  }

  __device__ __forceinline__
  void M2M(const int begin,
	   const int end,
	   const fvec4 Xi,
	   fvec4 * sourceCenter,
	   fvec4 * Multipole,
	   fvecP & Mi) {
    for (int i=begin; i<end; i++) {
      fvecP Mj = *(fvecP*)&Multipole[NVEC4*i];
      fvec4 Xj = sourceCenter[i];
      fvec3 dX = make_fvec3(Xi - Xj);
      fvecP C;
      C[0] = 1;
      Kernels<0,0,P-1>::power(C,dX);
      for (int j=0; j<NTERM; j++) Mi[j] += C[j] * Mj[0];
      Kernels<0,0,P-1>::M2M(Mi,C,Mj);
    }
  }

  __device__ __forceinline__
  fvec4 P2P(fvec4 acc,
	    const fvec3 pos_i,
	    const fvec3 pos_j,
	    const float q_j,
	    const float EPS2) {
    fvec3 dX = pos_j - pos_i;
    const float R2 = norm(dX) + EPS2;
    const float invR = rsqrtf(R2);
    const float invR2 = invR * invR;
    const float invR1 = q_j * invR;
    dX *= invR1 * invR2;
    acc[0] -= invR1;
    acc[1] += dX[0];
    acc[2] += dX[1];
    acc[3] += dX[2];
    return acc;
  }

#if 1
  __device__ __forceinline__
  fvec4 M2P(fvec4 acc,
	    const fvec3 & pos_i,
	    const fvec3 & pos_j,
	    const float * __restrict__ M,
	    float EPS2) {
    fvec3 dX = pos_i - pos_j;
    const float R2 = norm(dX) + EPS2;
    const float invR = rsqrtf(R2);
    const float invR2 = -invR * invR;
    const float invR1 = M[0] * invR;
    const float invR3 = invR2 * invR1;
    const float invR5 = 3 * invR2 * invR3;
    const float invR7 = 5 * invR2 * invR5;
    const float q11 = M[4];
    const float q12 = 0.5f * M[5];
    const float q13 = 0.5f * M[6];
    const float q22 = M[7];
    const float q23 = 0.5f * M[8];
    const float q33 = M[9];
    const float q = q11 + q22 + q33;
    fvec3 qR;
    qR[0] = q11 * dX[0] + q12 * dX[1] + q13 * dX[2];
    qR[1] = q12 * dX[0] + q22 * dX[1] + q23 * dX[2];
    qR[2] = q13 * dX[0] + q23 * dX[1] + q33 * dX[2];
    const float qRR = qR[0] * dX[0] + qR[1] * dX[1] + qR[2] * dX[2];
    acc[0] -= invR1 + invR3 * q + invR5 * qRR;
    const float C = invR3 + invR5 * q + invR7 * qRR;
    acc[1] += C * dX[0] + 2 * invR5 * qR[0];
    acc[2] += C * dX[1] + 2 * invR5 * qR[1];
    acc[3] += C * dX[2] + 2 * invR5 * qR[2];
    return acc;
  }
#else
  __device__ __forceinline__
  fvec4 M2P(fvec4 acc,
	    const fvec3 pos_i,
	    const fvec3 pos_j,
	    const float * __restrict__ M,
	    float EPS2) {
    const float x = pos_i[0] - pos_j[0];
    const float y = pos_i[1] - pos_j[1];
    const float z = pos_i[2] - pos_j[2];
    const float R2 = x * x + y * y + z * z + EPS2;
    const float invR = rsqrtf(R2);
    const float invR2 = -invR * invR;
    float C[20];
    const float invR1 = M[0] * invR;
    C[0] = invR1;
    const float invR3 = invR2 * invR1;
    C[1] = x * invR3;
    C[2] = y * invR3;
    C[3] = z * invR3;
    const float invR5 = 3 * invR2 * invR3;
    float t = x * invR5;
    C[4] = x * t + invR3;
    C[5] = y * t;
    C[6] = z * t;
    t = y * invR5;
    C[7] = y * t + invR3;
    C[8] = z * t;
    C[9] = z * z * invR5 + invR3;
    const float invR7 = 5 * invR2 * invR5;
    t = x * x * invR7;
    C[10] = x * (t + 3 * invR5);
    C[11] = y * (t +     invR5);
    C[12] = z * (t +     invR5);
    t = y * y * invR7;
    C[13] = x * (t +     invR5);
    C[16] = y * (t + 3 * invR5);
    C[17] = z * (t +     invR5);
    t = z * z * invR7;
    C[15] = x * (t +     invR5);
    C[18] = y * (t +     invR5);
    C[19] = z * (t + 3 * invR5);
    C[14] = x * y * z * invR7;
    acc[0] -= C[0]+M[4]*C[4] +M[5]*C[5] +M[6]*C[6] +M[7]*C[7] +M[8]*C[8] +M[9]*C[9];
    acc[1] += C[1]+M[4]*C[10]+M[5]*C[11]+M[6]*C[12]+M[7]*C[13]+M[8]*C[14]+M[9]*C[15];
    acc[2] += C[2]+M[4]*C[11]+M[5]*C[13]+M[6]*C[14]+M[7]*C[16]+M[8]*C[17]+M[9]*C[18];
    acc[3] += C[3]+M[4]*C[12]+M[5]*C[14]+M[6]*C[15]+M[7]*C[17]+M[8]*C[18]+M[9]*C[19];
    return acc;
  }
#endif
}
