#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  cudaThreadSynchronize();
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int   const N       = 1000000;
int   const THREADS = 64;
int   const NCRIT   = THREADS;
float const THETA   = 0.8;
float const EPS2    = 0.0001;

__global__ void direct(float4 *sourceGlob, float4 *targetGlob) {
  float3 d; 
  __shared__ float4 sourceShrd[THREADS];
  float4 target = sourceGlob[blockIdx.x * THREADS + threadIdx.x];
  target.w *= -rsqrtf(EPS2);
  for( int iblok=0; iblok<(N-1)/THREADS; iblok++) {
    __syncthreads();
    sourceShrd[threadIdx.x] = sourceGlob[iblok * THREADS + threadIdx.x];
    __syncthreads();
    for( int i=0; i<THREADS; i++ ) {
      d.x = target.x - sourceShrd[i].x;
      d.y = target.y - sourceShrd[i].y;
      d.z = target.z - sourceShrd[i].z;
      target.w += sourceShrd[i].w * rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z + EPS2);
    }
  } 
  int iblok = (N-1)/THREADS;
  __syncthreads();
  sourceShrd[threadIdx.x] = sourceGlob[iblok * THREADS + threadIdx.x];
  __syncthreads();
  for( int i=0; i<N - (iblok * THREADS); i++ ) {
    d.x = target.x - sourceShrd[i].x;
    d.y = target.y - sourceShrd[i].y;
    d.z = target.z - sourceShrd[i].z;
    target.w += sourceShrd[i].w * rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z + EPS2);
  }
  targetGlob[blockIdx.x * THREADS + threadIdx.x] = target;
} 

__device__ void multipole(int i, float4 &target, float *multipShrd) {
  float R,R3,R5;
  float3 d;
  d.x = target.x - multipShrd[i*13+0];
  d.y = target.y - multipShrd[i*13+1];
  d.z = target.z - multipShrd[i*13+2];
  R = rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
  R3 = R * R * R;
  R5 = R3 * R * R;
  target.w += multipShrd[i*13+ 3] * R;
  target.w += multipShrd[i*13+ 4] * (-d.x * R3);
  target.w += multipShrd[i*13+ 5] * (-d.y * R3);
  target.w += multipShrd[i*13+ 6] * (-d.z * R3);
  target.w += multipShrd[i*13+ 7] * (3 * d.x * d.x * R5 - 1 * R3);
  target.w += multipShrd[i*13+ 8] * (3 * d.y * d.y * R5 - 1 * R3);
  target.w += multipShrd[i*13+ 9] * (3 * d.z * d.z * R5 - 1 * R3);
  target.w += multipShrd[i*13+10] * (3 * d.x * d.y * R5);
  target.w += multipShrd[i*13+11] * (3 * d.y * d.z * R5);
  target.w += multipShrd[i*13+12] * (3 * d.z * d.x * R5);
}

__global__ void kernel(int *offSrcGlob, float4 *sourceGlob, int *offMtpGlob, float *multipGlob, float4 *targetGlob) {
  int N = offSrcGlob[blockIdx.x+1]-offSrcGlob[blockIdx.x];
  int offset = offSrcGlob[blockIdx.x];
  float3 d;
  __shared__ float4 sourceShrd[THREADS];
  __shared__ float  multipShrd[13*THREADS];
  float4 target = targetGlob[blockIdx.x * THREADS + threadIdx.x];
  target.w *= -rsqrtf(EPS2);
  for( int iblok=0; iblok<(N-1)/THREADS; iblok++) {
    __syncthreads();
    sourceShrd[threadIdx.x] = sourceGlob[offset + iblok * THREADS + threadIdx.x];
    __syncthreads();
    for( int i=0; i<THREADS; i++ ) {
      d.x = target.x - sourceShrd[i].x;
      d.y = target.y - sourceShrd[i].y;
      d.z = target.z - sourceShrd[i].z;
      target.w += sourceShrd[i].w * rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z + EPS2);
    }
  }
  int iblok = (N-1)/THREADS;
  __syncthreads();
  sourceShrd[threadIdx.x] = sourceGlob[offset + iblok * THREADS + threadIdx.x];
  __syncthreads();
  for( int i=0; i<N - (iblok * THREADS); i++ ) {
    d.x = target.x - sourceShrd[i].x;
    d.y = target.y - sourceShrd[i].y;
    d.z = target.z - sourceShrd[i].z;
    target.w += sourceShrd[i].w * rsqrtf(d.x * d.x + d.y * d.y + d.z * d.z + EPS2);
  }
  N = offMtpGlob[blockIdx.x+1]-offMtpGlob[blockIdx.x];
  offset = offMtpGlob[blockIdx.x];
  for( int iblok=0; iblok<(N-1)/THREADS; iblok++) {
    int index = offset + iblok * THREADS + threadIdx.x;
    __syncthreads();
    for( int i=0; i<13; i++ )
      multipShrd[threadIdx.x*13+i] = multipGlob[index*13+i];
    __syncthreads();
    for( int i=0; i<THREADS; i++ ) {
      multipole(i,target,multipShrd);
    }
  }
  iblok = (N-1)/THREADS;
  int index = offset + iblok * THREADS + threadIdx.x;
  __syncthreads();
  for( int i=0; i<13; i++ )
    multipShrd[threadIdx.x*13+i] = multipGlob[index*13+i];
  __syncthreads();
  for( int i=0; i<N - (iblok * THREADS); i++ ) {
    multipole(i,target,multipShrd);
  }
  targetGlob[blockIdx.x * THREADS + threadIdx.x] = target;
}

struct cell {
  int nleaf,nchild,leaf[NCRIT];
  float xc,yc,zc,r;
  float multipole[10];
  cell *parent,*child[8];
};

void initialize(cell *C) {
  C->nleaf = C->nchild = 0;
  C->parent = NULL;
  for( int i=0; i<8; i++ ) C->child[i] = NULL;
  for( int i=0; i<10; i++ ) C->multipole[i] = 0;
}

void add_child(int octant, cell *C, cell *&CN) {
  ++CN;
  initialize(CN);
  CN->r  = C->r/2;
  CN->xc = C->xc+CN->r*((octant&1)*2-1);
  CN->yc = C->yc+CN->r*((octant&2)-1);
  CN->zc = C->zc+CN->r*((octant&4)/2-1);
  CN->parent = C;
  C->child[octant] = CN;
  C->nchild |= (1 << octant);
}

void split_cell(float *x, float *y, float *z, cell *C, cell *&CN) {
  for( int i=0; i<NCRIT; i++ ) {                             
    int l = C->leaf[i];
    int octant = (x[l] > C->xc) + ((y[l] > C->yc) << 1) + ((z[l] > C->zc) << 2);
    if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN);
    cell *CC = C->child[octant];
    CC->leaf[CC->nleaf++] = l;
    if( CC->nleaf >= NCRIT ) split_cell(x,y,z,CC,CN);
  } 
}

void getMultipole(cell *C, float *x, float *y, float *z, float *m, cell **twig, int &ntwig) {
  float dx,dy,dz;
  if( C->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ )
      if( C->nchild & (1 << c) ) getMultipole(C->child[c],x,y,z,m,twig,ntwig);
  } else {
    for( int l=0; l<C->nleaf; l++ ) {
      int j = C->leaf[l];
      dx = C->xc-x[j];
      dy = C->yc-y[j];
      dz = C->zc-z[j];
      C->multipole[0] += m[j];
      C->multipole[1] += m[j]*dx;
      C->multipole[2] += m[j]*dy;
      C->multipole[3] += m[j]*dz;
      C->multipole[4] += m[j]*dx*dx/2;
      C->multipole[5] += m[j]*dy*dy/2;
      C->multipole[6] += m[j]*dz*dz/2;
      C->multipole[7] += m[j]*dx*dy/2;
      C->multipole[8] += m[j]*dy*dz/2;
      C->multipole[9] += m[j]*dz*dx/2;
    }
    twig[ntwig] = C;
    ntwig++;
  }
}

void upwardSweep(cell *C, cell *P) {
  float dx,dy,dz;
  dx = P->xc-C->xc;
  dy = P->yc-C->yc;
  dz = P->zc-C->zc;
  P->multipole[0] += C->multipole[0];
  P->multipole[1] += C->multipole[1]+ dx*C->multipole[0];
  P->multipole[2] += C->multipole[2]+ dy*C->multipole[0];
  P->multipole[3] += C->multipole[3]+ dz*C->multipole[0];
  P->multipole[4] += C->multipole[4]+ dx*C->multipole[1]+dx*dx*C->multipole[0]/2;
  P->multipole[5] += C->multipole[5]+ dy*C->multipole[2]+dy*dy*C->multipole[0]/2;
  P->multipole[6] += C->multipole[6]+ dz*C->multipole[3]+dz*dz*C->multipole[0]/2;
  P->multipole[7] += C->multipole[7]+(dx*C->multipole[2]+   dy*C->multipole[1]+dx*dy*C->multipole[0])/2;
  P->multipole[8] += C->multipole[8]+(dy*C->multipole[3]+   dz*C->multipole[2]+dy*dz*C->multipole[0])/2;
  P->multipole[9] += C->multipole[9]+(dz*C->multipole[1]+   dx*C->multipole[3]+dz*dx*C->multipole[0])/2;
}

void preval(cell *CI, cell *CJ, int &nmtp, int &nsrc) {
  float dx,dy,dz,r;
  if( CJ->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ ) {
      if( CJ->nchild & (1 << c) ) {
        cell *CC = CJ->child[c];
        dx = CI->xc-CC->xc;
        dy = CI->yc-CC->yc;
        dz = CI->zc-CC->zc;
        r = sqrtf(dx*dx+dy*dy+dz*dz);
        if( CI->r+CC->r > THETA*r ) {
          preval(CI,CC,nmtp,nsrc);
        } else {
          nmtp += 13;
        }
      }
    }
  } else
    nsrc += CJ->nleaf;
}

void evaluate(cell *CI, cell *CJ, float *x, float *y, float *z, float *m, float *p,
              int &offSrc, float4 *sourceHost, int &offMtp, float *multipHost) {
  float dx,dy,dz,r;
  if( CJ->nleaf >= NCRIT ) {
    for( int c=0; c<8; c++ ) {
      if( CJ->nchild & (1 << c) ) {
        cell *CC = CJ->child[c];
        dx = CI->xc-CC->xc;
        dy = CI->yc-CC->yc;
        dz = CI->zc-CC->zc;
        r = sqrtf(dx*dx+dy*dy+dz*dz);
        if( CI->r+CC->r > THETA*r ) {
          evaluate(CI,CC,x,y,z,m,p,offSrc,sourceHost,offMtp,multipHost);
        } else {
          multipHost[offMtp*13+ 0] = CC->xc;
          multipHost[offMtp*13+ 1] = CC->yc;
          multipHost[offMtp*13+ 2] = CC->zc;
          for( int i=0; i<10; i++ )
            multipHost[offMtp*13+ i + 3] = CC->multipole[i];
          offMtp++;
        }
      }
    }
  } else {
    for( int lj=0; lj<CJ->nleaf; lj++ ) {
      int j = CJ->leaf[lj];
      sourceHost[offSrc].x = x[j];
      sourceHost[offSrc].y = y[j];
      sourceHost[offSrc].z = z[j];
      sourceHost[offSrc].w = m[j];
      offSrc++;
    }
  }
}

int main() {
  float *x,*y,*z,*m,*p,*pd;
  double tic,toc;
  x  = (float*)malloc(N*sizeof(float));
  y  = (float*)malloc(N*sizeof(float));
  z  = (float*)malloc(N*sizeof(float));
  m  = (float*)malloc(N*sizeof(float));
  p  = (float*)malloc(N*sizeof(float));
  pd = (float*)malloc(N*sizeof(float));
// Initialize
  for( int i=0; i<N; i++ ) {
    x[i] = rand()/(1.+RAND_MAX);
    y[i] = rand()/(1.+RAND_MAX);
    z[i] = rand()/(1.+RAND_MAX);
    m[i] = 1.0/N;
  }
// Direct summation
/*
  float dx,dy,dz,r;
  for( int i=0; i<N; i++ ) {
    float pp = - m[i] / sqrtf(EPS2);
    for( int j=0; j<N; j++ ) {
      dx = x[i]-x[j];
      dy = y[i]-y[j];
      dz = z[i]-z[j];
      r = sqrtf(dx*dx+dy*dy+dz*dz+EPS2);
      pp += m[j] / r;
    }
    pd[i] = pp;
  }
*/
// Set root cell
  cell *C0;
  C0 = (cell*)malloc(N*sizeof(cell));
  initialize(C0);
  C0->xc = C0->yc = C0->zc = C0->r = 0.5;
// Build tree
  cell *CN = C0;
  for( int i=0; i<N; i++ ) {
    cell *C = C0;
    while( C->nleaf >= NCRIT ) {
      C->nleaf++;
      int octant = (x[i] > C->xc) + ((y[i] > C->yc) << 1) + ((z[i] > C->zc) << 2);
      if( !(C->nchild & (1 << octant)) ) add_child(octant,C,CN);
      C = C->child[octant];
    }
    C->leaf[C->nleaf++] = i;
    if( C->nleaf >= NCRIT ) split_cell(x,y,z,C,CN);
  }
// Multipole expansion
  int ntwig=0;
  cell **twig;
  twig = (cell**)malloc(N*sizeof(cell*));
  getMultipole(C0,x,y,z,m,twig,ntwig);
// Upward translation
  for( cell *C=CN; C!=C0; --C ) {
    cell *P = C->parent;
    upwardSweep(C,P);
  }
// Evaluate expansion
  int Nround = ntwig * THREADS;
  int ntgt = Nround;
  int noff = ntwig + 1;
  int    *offSrcHost,*offSrcDevc;
  int    *offMtpHost,*offMtpDevc;
  float4 *sourceHost,*sourceDevc;
  float4 *targetHost,*targetDevc;
  float  *multipHost,*multipDevc;
// Allocate memory on host and device
  int nmtp = 0, nsrc = 0;
  for( int t=0; t<ntwig; t++ ) {
    cell *CI = twig[t];
    cell *CJ = C0;
    preval(CI,CJ,nmtp,nsrc);
  }
  printf("%d %d\n",ntgt,nsrc);

  offSrcHost = (int   *)     malloc( noff*sizeof(int) );
  offMtpHost = (int   *)     malloc( noff*sizeof(int) );
  sourceHost = (float4*)     malloc( nsrc*sizeof(float4) );
  targetHost = (float4*)     malloc( ntgt*sizeof(float4) );
  multipHost = (float *)     malloc( nmtp*sizeof(float ) );
  cudaMalloc(  (void**) &offSrcDevc, noff*sizeof(int) );
  cudaMalloc(  (void**) &offMtpDevc, noff*sizeof(int) );
  cudaMalloc(  (void**) &sourceDevc, nsrc*sizeof(float4) );
  cudaMalloc(  (void**) &targetDevc, ntgt*sizeof(float4) );
  cudaMalloc(  (void**) &multipDevc, nmtp*sizeof(float ) );
  for( int i=0; i<N; i++ ) p[i] = 0;
  int offSrc = 0, offMtp = 0;
  tic = get_time();
  for( int t=0; t<ntwig; t++ ) {
    cell *CI = twig[t];
    cell *CJ = C0;
    offSrcHost[t] = offSrc;
    offMtpHost[t] = offMtp;
    for( int l=0; l<CI->nleaf; l++ ) {
      int i = CI->leaf[l];
      targetHost[t*THREADS+l].x = x[i];
      targetHost[t*THREADS+l].y = y[i];
      targetHost[t*THREADS+l].z = z[i];
      targetHost[t*THREADS+l].w = m[i];
    }
    evaluate(CI,CJ,x,y,z,m,p,offSrc,sourceHost,offMtp,multipHost);
  }
  toc = get_time();
  printf("buffer: %lf\n",toc-tic);
  offSrcHost[ntwig] = offSrc;
  offMtpHost[ntwig] = offMtp;
// Direct summation on device
  tic = get_time();
  cudaMemcpy(offSrcDevc,offSrcHost,noff*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(offMtpDevc,offMtpHost,noff*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(sourceDevc,sourceHost,nsrc*sizeof(float4),cudaMemcpyHostToDevice);
  cudaMemcpy(multipDevc,multipHost,nmtp*sizeof(float ),cudaMemcpyHostToDevice);
  cudaMemcpy(targetDevc,targetHost,ntgt*sizeof(float4),cudaMemcpyHostToDevice);
  toc = get_time();
  printf("memcpy: %lf\n",toc-tic);
  tic = get_time();
  kernel<<< Nround/THREADS, THREADS >>>(offSrcDevc,sourceDevc,offMtpDevc,multipDevc,targetDevc);
  toc = get_time();
  printf("kernel: %lf\n",toc-tic);
  cudaMemcpy(targetHost,targetDevc,ntgt*sizeof(float4),cudaMemcpyDeviceToHost);
// Compare results
  for( int t=0; t<ntwig; t++ ) {
    cell *CI = twig[t];
    for( int l=0; l<CI->nleaf; l++ ) {
      int i = CI->leaf[l];
      p[i] = targetHost[t*THREADS+l].w;
    }
  }
// Direct summation on device
  for( int i=0; i<N; i++ ) {
    sourceHost[i].x = x[i];
    sourceHost[i].y = y[i];
    sourceHost[i].z = z[i];
    sourceHost[i].w = m[i];
  }
  tic = get_time();
  cudaMemcpy        (sourceDevc,sourceHost,Nround*sizeof(float4),cudaMemcpyHostToDevice);
  direct<<< Nround/THREADS, THREADS >>>(sourceDevc,targetDevc);
  cudaMemcpy        (targetHost,targetDevc,Nround*sizeof(float4),cudaMemcpyDeviceToHost);
  toc = get_time();
  printf("direct: %lf\n",toc-tic);
// Compare results
  float err=0,rel=0;
  for( int i=0; i<N; i++ ) {
    float pp = targetHost[i].w;;
//    float pp = pd[i];
    err += (pp-p[i])*(pp-p[i]);
    rel += pp*pp;
  }
  printf("error : %f\n",sqrtf(err/rel));
}
