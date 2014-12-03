#include <mpi.h>
#include "kernels.h"

class Fmm : public Kernel {
protected:
  int gatherLevel;

private:
  inline void getIndex(int i, int *ix, real diameter) const {
    for_3d ix[d] = int((Jbodies[i][d] + R0 - X0[d]) / diameter);
  }

  void sort(real (*bodies)[4], float (*buffer)[4], int *Index, int *Index2, int *key) const {
    int Imax = key[0];
    int Imin = key[0];
    for( int i=0; i<numBodies; i++ ) {
      Imax = FMMMAX(Imax,key[i]);
      Imin = FMMMIN(Imin,key[i]);
    }
    int numBucket = Imax - Imin + 1;
    int *bucket = new int [numBucket];
    for( int i=0; i<numBucket; i++ ) bucket[i] = 0;
    for( int i=0; i<numBodies; i++ ) bucket[key[i]-Imin]++;
    for( int i=1; i<numBucket; i++ ) bucket[i] += bucket[i-1];
    for( int i=numBodies-1; i>=0; --i ) {
      bucket[key[i]-Imin]--;
      int inew = bucket[key[i]-Imin];
      Index2[inew] = Index[i];
      for_4d buffer[inew][d] = bodies[i][d];
    }
    delete[] bucket;
  }

public:
  void allocate(int N, int L) {
    maxLevel = L;
    numBodies = N;
    numCells = ((1 << 3 * (L + 1)) - 1) / 7;
    numLeafs = 1 << 3 * L;
    numSendCells = 64 * L + 48 * ((1 << (L + 1)) - 2) + 12 * (((1 << (2 * L + 2)) - 1) / 3 - 1);
    numSendLeafs = 8 + 12 * (1 << L) + 6 * (1 << (2 * L));
    numSendBodies = numSendLeafs * float(numBodies) / numLeafs * 2;
    Index = new int [2*numBodies];
    Index2 = new int [2*numBodies];
    Rank = new int [2*numBodies];
    Ibodies = new real [2*numBodies][4]();
    Jbodies = new real [2*numBodies+numSendBodies][4]();
    Multipole = new real [27*numCells][MTERM]();
    Local = new real [numCells][LTERM]();
    Leafs = new int [27*numLeafs][2]();
    globMultipole = new real [2][MTERM]();
    globLocal = new real [10][LTERM]();
    sendJbodies = new float [2*numBodies+numSendBodies][4]();
    recvJbodies = new float [2*numBodies+numSendBodies][4]();
    sendMultipole = new float [numSendCells][MTERM]();
    recvMultipole = new float [numSendCells][MTERM]();
    sendLeafs = new int [numSendLeafs][2]();
    recvLeafs = new int [numSendLeafs][2]();
  }

  void deallocate() {
    delete[] Index;
    delete[] Index2;
    delete[] Ibodies;
    delete[] Jbodies;
    delete[] Multipole;
    delete[] Local;
    delete[] Leafs;
    delete[] globMultipole;
    delete[] globLocal;
    delete[] sendJbodies;
    delete[] recvJbodies;
    delete[] sendMultipole;
    delete[] recvMultipole;
    delete[] sendLeafs;
    delete[] recvLeafs;
  }

  void partitioner(int level) {
    for_3d numPartition[0][d] = 1;
    numGlobCells = 0;
    globLevelOffset[0] = numGlobCells;
    numGlobCells = 1;
    gatherLevel = level;
    if(gatherLevel > maxGlobLevel) gatherLevel = maxGlobLevel;
  }

  void sortBodies() const {
    int *key = new int [numBodies];
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    for( int i=0; i<numBodies; i++ ) {
      getIndex(i,ix,diameter);
      key[i] = getKey(ix,maxLevel);
    }
    sort(Jbodies,sendJbodies,Index,Index2,key);
    for( int i=0; i<numBodies; i++ ) {
      Index[i] = Index2[i];
      for_4d Jbodies[i][d] = sendJbodies[i][d];
    }
    delete[] key;
  }

  void buildTree() const {
    int rankOffset = 13 * numLeafs;
    for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
      Leafs[i][0] = Leafs[i][1] = 0;
    }
    real diameter = 2 * R0 / (1 << maxLevel);
    int ix[3] = {0, 0, 0};
    getIndex(0,ix,diameter);
    int ileaf = getKey(ix,maxLevel,false) + rankOffset;
    Leafs[ileaf][0] = 0;
    for( int i=0; i<numBodies; i++ ) {
      getIndex(i,ix,diameter);
      int inew = getKey(ix,maxLevel,false) + rankOffset;
      if( ileaf != inew ) {
        Leafs[ileaf][1] = Leafs[inew][0] = i;
        ileaf = inew;
      }
    }
    Leafs[ileaf][1] = numBodies;
    for( int i=rankOffset; i<numLeafs+rankOffset; i++ ) {
      //assert( Leafs[i][1] != Leafs[i][0] );
    }
  }
};
