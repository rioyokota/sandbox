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
  int i,j,N=1000;
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
    int level = 3;
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
  int ic = index[0], id = 1;
  int offset[N];
  offset[0] = 0;
  for (i=0; i<N; i++) {
    if (ic != index[i]) {
      offset[id] = i;
      ic = index[i];
      printf("%d %d\n",id,i);
      id++;
    }
  }
}
