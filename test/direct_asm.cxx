#include <assert.h>

// output
#define AX   "%xmm0"
#define AY   "%xmm1"
#define AZ   "%xmm2"
#define PHI  "%xmm3"
// source
#define XJ   "%xmm4"
#define YJ   "%xmm5"
#define ZJ   "%xmm6"
#define MJ   "%xmm7"
// temporary
#define RINV "%xmm8"
#define X2   "%xmm9"
#define Y2   "%xmm10"
#define Z2   "%xmm11"
// target
#define XI   "%xmm12"
#define YI   "%xmm13"
#define ZI   "%xmm14"
#define R2   "%xmm15"

#define XORPS(a, b) asm ("xorps "a","b);
#define LOADPS(mem, reg) asm ("movaps %0, %"reg::"m"(mem));
#define STORPS(reg, mem) asm ("movaps %"reg", %0"::"m"(mem));
#define MOVAPS(src, dst) asm ("movaps "src","dst);
#define BCAST0(reg) asm ("shufps $0x00,"reg","reg);
#define BCAST1(reg) asm ("shufps $0x55,"reg","reg);
#define BCAST2(reg) asm ("shufps $0xaa,"reg","reg);
#define BCAST3(reg) asm ("shufps $0xff,"reg","reg);
#define MULPS(src, dst) asm ("mulps "src","dst);
#define ADDPS(src, dst) asm ("addps "src","dst);
#define SUBPS(src, dst) asm ("subps "src","dst);
#define RSQRTPS(src, dst) asm ("rsqrtps "src","dst);

struct float4 {
  float x;
  float y;
  float z;
  float w;
};

struct float16 {
  float x[4];
  float y[4];
  float z[4];
  float w[4];
};

void P2PasmKernel(float16 *target, const float4 *source, const int &n){
  assert(((unsigned long)source & 15) == 0);
  assert(((unsigned long)target & 15) == 0);

  XORPS(AX, AX);               // AX = 0
  XORPS(AY, AY);               // AY = 0
  XORPS(AZ, AZ);               // AZ = 0
  XORPS(PHI, PHI);             // PHI = 0

  LOADPS(*target->x, XI);      // XI = target->x
  LOADPS(*target->y, YI);      // YI = target->y
  LOADPS(*target->z, ZI);      // ZI = target->z
  LOADPS(*target->w, R2);      // R2 = target->w

  LOADPS(source->x, MJ);       // MJ = source->x,y,z,w
  MOVAPS(MJ, X2);              // X2 = MJ
  MOVAPS(MJ, Y2);              // Y2 = MJ
  MOVAPS(MJ, Z2);              // Z2 = MJ

  BCAST0(X2);                  // X2 = source->x
  SUBPS(XI, X2);               // X2 = X2 - XI
  BCAST1(Y2);                  // Y2 = source->y
  SUBPS(YI, Y2);               // Y2 = Y2 - YI
  BCAST2(Z2);                  // Z2 = source->z
  SUBPS(ZI, Z2);               // Z2 = Z2 - ZI
  BCAST3(MJ);                  // MJ = source->w

  MOVAPS(X2, XJ);              // XJ = X2
  MULPS(X2, X2);               // X2 = X2 * X2
  ADDPS(X2, R2);               // R2 = R2 + X2

  MOVAPS(Y2, YJ);              // YJ = Y2
  MULPS(Y2, Y2);               // Y2 = Y2 * Y2
  ADDPS(Y2, R2);               // R2 = R2 + Y2

  MOVAPS(Z2, ZJ);              // ZJ = Z2
  MULPS(Z2, Z2);               // Z2 = Z2 * Z2
  ADDPS(Z2, R2);               // R2 = R2 + Z2

  LOADPS(source[1].x, X2);     // X2 = next source->x,y,z,w
  MOVAPS(X2, Y2);              // Y2 = X2
  MOVAPS(X2, Z2);              // Z2 = X2
  for( int j=0; j<n; j++ ) {
    RSQRTPS(R2, RINV);         // RINV = rsqrt(R2)       1
    source++;
    LOADPS(*target->w, R2);    // R2 = target->w
    BCAST0(X2);                // X2 = source->x
    SUBPS(XI, X2);             // X2 = X2 - XI           2
    BCAST1(Y2);                // Y2 = source->y
    SUBPS(YI, Y2);             // Y2 = Y2 - YI           3
    BCAST2(Z2);                // Z2 = source->z
    SUBPS(ZI, Z2);             // Z2 = Z2 - ZI           4

    MULPS(RINV, MJ);           // MJ = MJ * RINV         5
    ADDPS(MJ, PHI);            // PHI = PHI + MJ         6
    MULPS(RINV, RINV);         // RINV = RINV * RINV     7
    MULPS(MJ, RINV);           // RINV = MJ * RINV       8
    LOADPS(source->x, MJ);     // MJ = source->x,y,z,w
    BCAST3(MJ);                // MJ = source->w

    MULPS(RINV, XJ);           // XJ = XJ * RINV         9
    ADDPS(XJ, AX);             // AX = AX + XJ          10
    MOVAPS(X2, XJ);            // XJ = X2
    MULPS(X2, X2);             // X2 = X2 * X2          11
    ADDPS(X2, R2);             // R2 = R2 + X2          12
    LOADPS(source[1].x, X2);   // X2 = next source->x,y,z,w

    MULPS(RINV, YJ);           // YJ = YJ * RINV        13
    ADDPS(YJ, AY);             // AY = AY + YJ          14
    MOVAPS(Y2, YJ);            // YJ = Y2
    MULPS(Y2, Y2);             // Y2 = Y2 * Y2          15
    ADDPS(Y2, R2);             // R2 = R2 + Y2          16
    MOVAPS(X2, Y2);            // Y2 = X2

    MULPS(RINV, ZJ);           // ZJ = ZJ * RINV        17
    ADDPS(ZJ, AZ);             // AZ = AZ + ZJ          18
    MOVAPS(Z2, ZJ);            // ZJ = Z2
    MULPS(Z2, Z2);             // Z2 = Z2 * Z2          19
    ADDPS(Z2, R2);             // R2 = R2 + Z2          20
    MOVAPS(X2, Z2);            // Z2 = X2
  }
  STORPS(AX,  *target->x);     // target->x = AX
  STORPS(AY,  *target->y);     // target->y = AY
  STORPS(AZ,  *target->z);     // target->z = AZ
  STORPS(PHI, *target->w);     // target->w = PHI
}

void P2Pasm(float4 *target, float4 *source, int ni, int nj, float eps2) {
#pragma omp parallel for
  for( int base=0; base<ni; base+=4 ) {
    float16 target4;
    for( int i=0; i<4; i++ ) {
      target4.x[i] = source[base+i].x;
      target4.y[i] = source[base+i].y;
      target4.z[i] = source[base+i].z;
      target4.w[i] = eps2;
    }
    P2PasmKernel(&target4, source, nj);
    for( int i=0; i<4; i++ ) {
      target[base+i].x = target4.x[i];
      target[base+i].y = target4.y[i];
      target[base+i].z = target4.z[i];
      target[base+i].w = target4.w[i];
    }
  }
}

#undef AX
#undef AY
#undef AZ
#undef PHI
#undef XJ
#undef YJ
#undef ZJ
#undef MJ
#undef RINV
#undef X2
#undef Y2
#undef Z2
#undef XI
#undef YI
#undef ZI
#undef R2
#undef XORPS
#undef LOADPS
#undef STORPS
#undef MOVAPS
#undef BCAST0
#undef BCAST1
#undef BCAST2
#undef BCAST3
#undef MULPS
#undef ADDPS
#undef SUBPS
#undef RSQRTPS
