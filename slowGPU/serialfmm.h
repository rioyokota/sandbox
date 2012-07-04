#ifndef serialfmm_h
#define serialfmm_h
#include "evaluator.h"

class SerialFMM : public Evaluator {
protected:
  int MAXLEVEL;

private:
  inline void setMorton(int *key) {
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int b=0; b<numBodies; b++ ) {
      int ix = int((Jbodies[b][0] + R0 - X0[0]) / d);
      int iy = int((Jbodies[b][1] + R0 - X0[1]) / d);
      int iz = int((Jbodies[b][2] + R0 - X0[2]) / d);
      int id = 0;
      for( int l=0; l<MAXLEVEL; ++l ) {
        id += ix % 2 << (3 * l);
        id += iy % 2 << (3 * l + 1);
        id += iz % 2 << (3 * l + 2);
        ix >>= 1;
        iy >>= 1;
        iz >>= 1;
      }
      key[b] = id;
    }
  }

  void sortBodies(int *key) {
    int Imax = key[0];
    int Imin = key[0];
    for( int i=0; i<numBodies; i++ ) {
      Imax = std::max(Imax,key[i]);
      Imin = std::min(Imin,key[i]);
    }
    int numBucket = Imax - Imin + 1;
    int *bucket = new int [numBucket];
    int *buffer = new int [numBodies];
    for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
    for( int i=0; i<numBodies; i++ ) bucket[key[i]-Imin]++;
    for( int i=1; i<numBucket; i++ ) bucket[i] += bucket[i-1];
    for( int i=numBodies-1; i>=0; --i ) {
      bucket[key[i]-Imin]--;
      int inew = bucket[key[i]-Imin];
      buffer[inew] = key[i];
      for( int d=0; d<4; d++ ) Ibodies[inew][d] = Jbodies[i][d];
    }
    for( int i=0; i<numBodies; i++ ) {
      key[i] = buffer[i];
      for( int d=0; d<4; d++ ) Jbodies[i][d] = Ibodies[i][d];
      for( int d=0; d<4; d++ ) Ibodies[i][d] = 0;
    }
    delete[] bucket;
    delete[] buffer;
  }

  inline void initCell(Cell &cell, int child, int b, real diameter) {
    cell.NCHILD = 0;
    cell.NCLEAF = 0;
    cell.NDLEAF = 0;
    cell.CHILD  = child;
    cell.LEAF   = b;
    int ix = int((Jbodies[b][0] + R0 - X0[0]) / diameter);
    int iy = int((Jbodies[b][1] + R0 - X0[1]) / diameter);
    int iz = int((Jbodies[b][2] + R0 - X0[2]) / diameter);
    cell.X[0]   = diameter * (ix + .5) + X0[0] - R0;
    cell.X[1]   = diameter * (iy + .5) + X0[1] - R0;
    cell.X[2]   = diameter * (iz + .5) + X0[2] - R0;
    cell.R      = diameter * .5;
  }

  void buildBottom(int *key) {
    int I = -1;
    Cells.alloc(1 << (3 * MAXLEVEL + 1));
    int c = -1;
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int b=0; b<numBodies; b++ ) {
      int IC = key[b];
      if( IC != I ) {
        Cell cell;
        initCell(cell,0,b,d);
        cell.ICELL = IC;
        c++;
        Cells[c] = cell;
        I = IC;
      }
      Cells[c].NCLEAF++;
      Cells[c].NDLEAF++;
    }
    numCells = c+1;
  }

protected:
  void setDomain() {
    X0 = R0 = .5;
  }

  void buildTree() {
    int *key = new int [numBodies];
    setMorton(key);
    sortBodies(key);
    buildBottom(key);
    linkTree(key);
    delete[] key;
  }

  void linkTree(int *key) {
    int begin = 0, end = numCells;
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int l=0; l<MAXLEVEL; ++l ) {
      int div = (8 << (3 * l));
      int I = -1;
      int p = end - 1;
      d *= 2;
      for( int c=begin; c<end; ++c ) {
        int IC = key[Cells[c].LEAF] / div;
        if( IC != I ) {
          Cell cell;
          initCell(cell,c,Cells[c].LEAF,d);
          cell.ICELL = IC;
          p++;
          Cells[p] = cell;
          I = IC;
        }
        Cells[p].NCHILD++;
        Cells[p].NDLEAF += Cells[c].NDLEAF;
        Cells[c].PARENT = p;
      }
      begin = end;
      end = p + 1;
    }
    numCells = end;
  }

public:
  SerialFMM(int N) {
    Ibodies.alloc(N);
    Jbodies.alloc(N);
  }

  void dataset(int N) {
    numBodies = N;
    MAXLEVEL =  numBodies >= NCRIT ? 1 + int(log(numBodies / NCRIT)/M_LN2/3) : 0;
    srand48(0);
    for( int b=0; b<numBodies; b++ ) {
      for( int d=0; d<3; ++d ) {
        Jbodies[b][d] = drand48();
      }
      Jbodies[b][3] = 1. / numBodies;
    }
  }

  void bottomup() {
    double tic, toc;
    tic = getTime();
    setDomain();
    buildTree();
    toc = getTime();
    if( printNow ) printf("Tree                 : %lf\n",toc-tic);
    tic = getTime();
    Multipole.alloc(numCells);
    Local.alloc(numCells);
    upwardPass();
    toc = getTime();
    if( printNow ) printf("Upward pass          : %lf\n",toc-tic);
  }

  void evaluate() {
    double tic, toc;
    std::queue<int> Queue;
    Queue.push(numCells-1);
    while( Queue.size() < 100 ) {
      Cell *C = Cells.host() + Queue.front();
      if( C->NCHILD == 0 ) return;
      Queue.pop();
      for( int c=C->CHILD; c<C->CHILD+C->NCHILD; c++ ) {
        Queue.push(c);
      }
    }
    cudaVec<int> Branch;
    Branch.alloc(Queue.size());
    int c = 0;
    while( !Queue.empty() ) {
      Branch[c++] = Queue.front();
      Queue.pop();
    }
    Ibodies.h2d();
    Jbodies.h2d();
    Cells.h2d();
    Multipole.h2d();
    Local.h2d();
    Branch.h2d();
    tic = getTime();
    traverse<<<Branch.size(),NCRIT>>>(numCells,Branch.devc(),Ibodies.devc(),Jbodies.devc(),Cells.devc(),Multipole.devc(),Local.devc());
    Ibodies.d2h();
    Local.d2h();
    toc = getTime();
    if( printNow ) printf("Traverse             : %lf\n",toc-tic);
    tic = getTime();
    downwardPass();
    toc = getTime();
    if( printNow ) printf("Downward pass        : %lf\n",toc-tic);
  }

};

#endif
