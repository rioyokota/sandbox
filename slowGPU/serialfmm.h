#ifndef serialfmm_h
#define serialfmm_h
#include "evaluator.h"

class SerialFMM : public Evaluator {
protected:
  int MAXLEVEL;
  int *Index;

private:
  int getMaxLevel() {
    return numBodies >= NCRIT ? 1 + int(log(numBodies / NCRIT)/M_LN2/3) : 0;
  }

  inline void setMorton() {
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
      Index[b] = id;
    }
  }

  void sortBodies() {
    int Imax = Index[0];
    int Imin = Index[0];
    for( int i=0; i<numBodies; i++ ) {
      Imax = std::max(Imax,Index[i]);
      Imin = std::min(Imin,Index[i]);
    }
    int numBucket = Imax - Imin + 1;
    int *bucket = new int [numBucket];
    int *buffer = new int [numBodies];
    for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
    for( int i=0; i<numBodies; i++ ) bucket[Index[i]-Imin]++;
    for( int i=1; i<numBucket; i++ ) bucket[i] += bucket[i-1];
    for( int i=numBodies-1; i>=0; --i ) {
      bucket[Index[i]-Imin]--;
      int inew = bucket[Index[i]-Imin];
      buffer[inew] = Index[i];
      for( int d=0; d<4; d++ ) Ibodies[inew][d] = Jbodies[i][d];
    }
    for( int i=0; i<numBodies; i++ ) {
      Index[i] = buffer[i];
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

  void buildBottom() {
    int I = -1;
    if( C0 != NULL ) delete[] C0;
    C0 = new Cell [1 << (3 * MAXLEVEL + 1)];
    int c = -1;
    float d = 2 * R0 / (1 << MAXLEVEL);
    for( int b=0; b<numBodies; b++ ) {
      int IC = Index[b];
      if( IC != I ) {
        Cell cell;
        initCell(cell,0,b,d);
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
  void setDomain() {
    MAXLEVEL = getMaxLevel();
    X0 = R0 = .5;
  }

  void buildTree() {
    setMorton();
    sortBodies();
    buildBottom();
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
        int IC = Index[C0[c].LEAF] / div;
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

  void dataset(int N) {
    numBodies = N;
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
    CellPair pair(ROOT,ROOT);
    PairStack pairStack;
    pairStack.push(pair);
    traverse(pairStack);
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
