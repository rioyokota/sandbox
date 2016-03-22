#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void getIndex(int *index, int iX[2], int level){
  int d,l;
  *index = 0;
  for (l=0; l<level; l++) {
    for (d=0; d<2; d++) {
      *index += iX[d] % 2 << (2*l+1-d);
      iX[d] >>= 1;
    }
  }
}

void getIX(int index, int iX[2]) {
  int level = 0;
  int d = 0;
  iX[0] = iX[1] = 0;
  while( index > 0 ) {
    iX[1-d] += (index % 2) * (1 << level);
    index >>= 1;
    d = (d+1) % 2;
    if( d == 0 ) level++;
  }
}

void radixSort(int * key, int * value, int size) {
  const int bitStride = 8;
  const int stride = 1 << bitStride;
  const int mask = stride - 1;
  int i, maxKey = 0;
  int bucket[stride];
  int * buffer = (int*) malloc(size*sizeof(int));
  int * permutation = (int*) malloc(size*sizeof(int));
  for (i=0; i<size; i++)
    if (key[i] > maxKey)
      maxKey = key[i];
  while (maxKey > 0) {
    for (i=0; i<stride; i++)
      bucket[i] = 0;
    for (i=0; i<size; i++)
      bucket[key[i] & mask]++;
    for (i=1; i<stride; i++)
      bucket[i] += bucket[i-1];
    for (i=size-1; i>=0; i--)
      permutation[i] = --bucket[key[i] & mask];
    for (i=0; i<size; i++)
      buffer[permutation[i]] = value[i];
    for (i=0; i<size; i++)
      value[i] = buffer[i];
    for (i=0; i<size; i++)
      buffer[permutation[i]] = key[i];
    for (i=0; i<size; i++)
      key[i] = buffer[i] >> bitStride;
    maxKey >>= bitStride;
  }
  free(buffer);
  free(permutation);
}

int main() {
  int i,j,N = 1000;
  int level = 3;
  int index[N];
  double x[N], y[N], u[N], q[N];
  srand48(1);
  for (i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    u[i] = 0;
    q[i] = 1;
  }
  int iX[2],perm[N],index2[N];
  double x2[N],y2[N];
  for (i=0; i<N; i++) {
    iX[0] = x[i] * 8;
    iX[1] = y[i] * 8;
    getIndex(&index[i], iX, level);
    perm[i] = i;
    index2[i] = index[i];
    x2[i] = x[i];
    y2[i] = y[i];
  }
  radixSort(index,perm,N);
  for (i=0; i<N; i++) {
    index[i] = index2[perm[i]];
    x[i] = x2[perm[i]];
    y[i] = y2[perm[i]];
  }
  int ic = index[0], ncells = 1;
  int offset[N];
  int leafs = 1<<(2*level);
  int map[leafs];
  for (i=0; i<leafs; i++) {
    map[i] = -1;
  }
  offset[0] = 0;
  map[ic] = 0;
  for (i=0; i<N; i++) {
    if (ic != index[i]) {
      offset[ncells] = i;
      map[index[i]] = ncells;
      ic = index[i];
      ncells++;
    }
  }
  offset[ncells] = N;
  int maxcells = 0;
  int levelOffset[4];
  levelOffset[0] = 0;
  for (i=0; i<=level; i++) {
    maxcells += 1 << (2*i);
    levelOffset[i+1] = maxcells;
  }
  double Multipole[maxcells];
  double Local[maxcells];
  for (i=0; i<maxcells; i++) {
    Multipole[i] = 0;
    Local[i] = 0;
  }
  printf("P2M\n");
  // P2M
  for (i=0; i<leafs; i++) {
    if (map[i] >=0) {
      for (j=offset[map[i]]; j<offset[map[i]+1]; j++) {
	Multipole[index[j]+levelOffset[level]] += q[j];
      }
    }
  }
  printf("M2M\n");
  // M2M
  int l;
  for (l=level; l>0; l--) {
    int nc = 1 << (2*l);
    for (i=0; i<nc; i++) {
      int ip = i/4 + levelOffset[l-1];
      Multipole[ip] += Multipole[i+levelOffset[l]];
    }
  }
  printf("M2L\n");
  // M2L
  for (l=2; l<=level; l++) {
    int nc = 1 << (2*l);
    for (i=0; i<nc; i++) {
      getIX(i/4, iX);
      int ix, iy;
      int ixmin = -1, iymin = -1;
      int ixmax =  1, iymax =  1;
      if(iX[0]==0) ixmin = 0;
      if(iX[1]==0) iymin = 0;
      if(iX[0]==(1<<(l-1))-1) ixmax = 0;
      if(iX[1]==(1<<(l-1))-1) iymax = 0;
      for (ix=ixmin; ix<=ixmax; ix++) {
	for (iy=iymin; iy<=iymax; iy++) {
	  int iX2[2],jX[2];
	  iX2[0] = iX[0]+ix;
	  iX2[1] = iX[1]+iy;
	  getIndex(&ic, iX2, l-1);
	  int jc;
	  for (jc=0; jc<4; jc++) {
	    getIX(ic*4+jc,jX);
	    if(abs(iX[0]-jX[0])>1||abs(iX[1]-jX[1])>1) {
	      double dx = (iX[0] - jX[0] + 0.0) / (1 << l);
	      double dy = (iX[1] - jX[1] + 0.0) / (1 << l);
	      double r = sqrt(dx*dx+dy*dy);
	      getIndex(&j, jX, l);
	      Local[i+levelOffset[l]] += Multipole[j+levelOffset[l]] / r;
	    }
	  }
	}
      }
    }
  }
  printf("L2L\n");
  // L2L
  for (l=2; l<=level; l++) {
    int nc = 1 << (2*l);
    for (i=0; i<nc; i++) {
      int ip = i/4 + levelOffset[l-1];
      Local[i+levelOffset[l]] += Local[ip];
    }
  }
  printf("L2P\n");
  // L2P
  for (i=0; i<leafs; i++) {
    if (map[i] >=0) {
      for (j=offset[map[i]]; j<offset[map[i]+1]; j++) {
	u[j] += Local[index[j]+levelOffset[level]];
      }
    }
  }
  printf("P2P\n");
  // P2P
  for (ic=0; ic<leafs; ic++) {
    if (map[ic] >= 0) {
      getIX(ic,iX);
      int ix, iy;
      int ixmin = -1, iymin = -1;
      int ixmax =  1, iymax =  1;
      if(iX[0]==0) ixmin = 0;
      if(iX[1]==0) iymin = 0;
      if(iX[0]==(1<<level)-1) ixmax = 0;
      if(iX[1]==(1<<level)-1) iymax = 0;
      for (ix=ixmin; ix<=ixmax; ix++) {
	for (iy=iymin; iy<=iymax; iy++) {
	  int iX2[2],jX[2];
	  iX2[0] = iX[0]+ix;
	  iX2[1] = iX[1]+iy;
	  int jc;
	  if (map[jc] >=0) {
	    getIndex(&jc,iX2,level);
	    for (i=offset[map[ic]]; i<offset[map[ic]+1]; i++) {
	      for (j=offset[map[jc]]; j<offset[map[jc]+1]; j++) {
		double dx = x[i] - x[j];
		double dy = y[i] - y[j];
		double r = sqrt(dx*dx+dy*dy);
		if(r!=0) u[i] += q[j] / r;
	      }
	    }
	  }
	}
      }
    }
  }
  // Check answer
  printf("Direct\n");
  for (i=0; i<N; i++) {
    double ud = 0;
    for (j=0; j<N; j++) {
      double dx = x[i] - x[j];
      double dy = y[i] - y[j];
      double r = sqrt(dx*dx+dy*dy);
      if(r!=0) ud += q[j] / r;
    }
    printf("%lf %lf\n",u[i],ud);
  }
}

