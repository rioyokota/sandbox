#ifndef serialfmm_h
#define serialfmm_h
#include "evaluator.h"

class SerialFMM : public Evaluator {
protected:
  int MAXLEVEL;

private:
  int getMaxLevel(Bodies &bodies) {
    const long N = bodies.size();
    int level;
    level = N >= NCRIT ? 1 + int(log(N / NCRIT)/M_LN2/3) : 0;
    return level;
  }

  inline void setMorton(Bodies &bodies) {
    float d = 2 * R0 / (1 << MAXLEVEL);
    int b = 0;
    for( Body *B=&*bodies.begin(); B<&*bodies.end(); ++B,++b ) {
      int ix = int((B->X[0] + R0 - X0[0]) / d);
      int iy = int((B->X[1] + R0 - X0[1]) / d);
      int iz = int((B->X[2] + R0 - X0[2]) / d);
      int id = 0;
      for( int l=0; l<MAXLEVEL; ++l ) {
        id += ix % 2 << (3 * l);
        id += iy % 2 << (3 * l + 1);
        id += iz % 2 << (3 * l + 2);
        ix >>= 1;
        iy >>= 1;
        iz >>= 1;
      }
      B->ICELL = id;
      Index[b] = id;
    }
  }

  inline void radixSort() {
    const int bitStride = 8;
    const int stride = 1 << bitStride;
    const int stride1 = stride - 1;
    int *buffer = new int [numBodies];
    int imax = 0;
    for( int i=0; i<numBodies; i++ )
      if( Index[i] > imax ) imax = Index[i];
    int exp = 0;
    while( (imax >> exp) > 0 ) {
      int bucket[stride] = {0};
      for( int i=0; i<numBodies; i++ )
        bucket[(Index[i] >> exp) & stride1]++;
      for( int i=1; i<stride; i++ )
        bucket[i] += bucket[i-1];
      for( int i=numBodies-1; i>=0; i-- )
        buffer[--bucket[(Index[i] >> exp) & stride1]] = Index[i];
      for( int i=0; i<numBodies; i++ )
        Index[i] = buffer[i];
      exp += bitStride;
    }
    delete[] buffer;
  }

  inline void initCell(Cell &cell, int child, int LEAF, real diameter) {
    cell.NCHILD = 0;
    cell.NCLEAF = 0;
    cell.NDLEAF = 0;
    cell.CHILD  = child;
    cell.LEAF   = LEAF;
    int ix = int(((B0+LEAF)->X[0] + R0 - X0[0]) / diameter);
    int iy = int(((B0+LEAF)->X[1] + R0 - X0[1]) / diameter);
    int iz = int(((B0+LEAF)->X[2] + R0 - X0[2]) / diameter);
    cell.X[0]   = diameter * (ix + .5) + X0[0] - R0;
    cell.X[1]   = diameter * (iy + .5) + X0[1] - R0;
    cell.X[2]   = diameter * (iz + .5) + X0[2] - R0;
    cell.R      = diameter * .5;
  }

  void buildBottom(Bodies &bodies) {
    B0 = &bodies.front();
    int I = -1;
    delete[] C0;
    C0 = new Cell [1 << (3 * MAXLEVEL + 1)];
    int c = -1;
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( Body *B=&*bodies.begin(); B<&*bodies.end(); ++B ) {
      int IC = B->ICELL;
      if( IC != I ) {
        Cell cell;
        initCell(cell,0,B-B0,d);
        cell.ICELL = IC;
        c++;
        C0[c] = cell;
        I = IC;
      }
      C0[c].NCLEAF++;
      C0[c].NDLEAF++;
    }
    numCells = c+1;
  }

protected:
  void setDomain(Bodies &bodies) {
    MAXLEVEL = getMaxLevel(bodies);
    X0 = R0 = .5;
  }

  void buildTree(Bodies &bodies) {
    setMorton(bodies);
    Bodies buffer = bodies;
    sort(bodies.begin(),bodies.end());
    radixSort();
    buildBottom(bodies);
  }

  void linkTree() {
    int begin = 0, end = numCells;
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int l=0; l<MAXLEVEL; ++l ) {
      int div = (8 << (3 * l));
      int I = -1;
      int p = end - 1;
      d *= 2;
      for( int c=begin; c<end; ++c ) {
        Body *B = B0 + C0[c].LEAF;
        int IC = B->ICELL / div;
        if( IC != I ) {
          Cell cell;
          initCell(cell,c,C0[c].LEAF,d);
          cell.ICELL = IC;
          p++;
          C0[p] = cell;
          I = IC;
        }
        C0[p].NCHILD++;
        C0[p].NDLEAF += C0[c].NDLEAF;
        C0[c].PARENT = p;
      }
      begin = end;
      end = p + 1;
    }
    numCells = end;
    ROOT = C0 + numCells - 1;
  }

public:
  SerialFMM() {
    Index   = new int  [1000000];
    Ibodies = new real [1000000][4]();
    Jbodies = new real [1000000][4]();
  }
  ~SerialFMM() {
    delete[] Index;
    delete[] Ibodies;
    delete[] Jbodies;
  }

  void dataset(Bodies &bodies, int N) {
    numBodies = N;
    srand48(0);
    int b = 0;
    for( Body *B=&*bodies.begin(); B<&*bodies.end(); ++B,++b ) {      // Loop over bodies
      for( int d=0; d<3; ++d ) {
        B->X[d] = drand48();
        Jbodies[b][d] = B->X[d];
      }
      B->SRC = 1. / bodies.size();
      Jbodies[b][3] = 1. / bodies.size();
      B->TRG = 0;
      for( int d=0; d<4; ++d ) Ibodies[b][d] = 0;
    }
  }

  void bottomup(Bodies &bodies) {
    double tic, toc;
    tic = getTime();
    setDomain(bodies);
    buildTree(bodies);
    linkTree();
    toc = getTime();
    if( printNow ) printf("Tree                 : %lf\n",toc-tic);
    tic = getTime();
    Multipole = new real [numCells][MTERM]();
    Local = new real [numCells][LTERM]();
    upwardPass();
    toc = getTime();
    if( printNow ) printf("Upward pass          : %lf\n",toc-tic);
  }

  void evaluate() {
    double tic, toc;
    tic = getTime();
    Pair pair(ROOT,ROOT);
    PairQueue pairQueue;
    pairQueue.push_front(pair);
    traverse(pairQueue);
    toc = getTime();
    if( printNow ) printf("Traverse             : %lf\n",toc-tic);
    tic = getTime();
    downwardPass();
    toc = getTime();
    if( printNow ) printf("Downward pass        : %lf\n",toc-tic);
    delete[] Multipole;
    delete[] Local;
  }

};

#endif
