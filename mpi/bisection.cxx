#include "mpi.h"
#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <sys/time.h>
#include "types.h"

bool compareX(Body Bi, Body Bj) { return (Bi.X[0] < Bj.X[0]); }
bool compareY(Body Bi, Body Bj) { return (Bi.X[1] < Bj.X[1]); }
bool compareZ(Body Bi, Body Bj) { return (Bi.X[2] < Bj.X[2]); }

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char **argv) {
  int mpirank, mpisize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  int numBodies = 1000000;
  int numBins = 16;
  int numLevels = 0;
  int size = mpisize - 1;
  while (size > 0) {
    size >>= 1;
    numLevels++;
  }
  int rankDispl[mpisize];
  int rankCount[mpisize];
  int rankColor[mpisize];
  int rankKey[mpisize];
  int rankMap[mpisize];
  int sendDispl[mpisize];
  int sendCount[mpisize];
  int scanHist[numBins];
  int localHist[numBins];
  int globalHist[numBins];
  Bounds rankBounds[mpisize];
  Bodies bodies(numBodies);
  Bodies buffer(numBodies);
  srand48(mpirank);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->X[0] = drand48();
    B->X[1] = drand48();
    B->X[2] = drand48();
  }
  Bounds localBounds;
  localBounds.Xmin = localBounds.Xmax = bodies.front().X;
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    localBounds.Xmin[0] = B->X[0] < localBounds.Xmin[0] ? B->X[0] : localBounds.Xmin[0];
    localBounds.Xmin[1] = B->X[1] < localBounds.Xmin[1] ? B->X[1] : localBounds.Xmin[1];
    localBounds.Xmin[2] = B->X[2] < localBounds.Xmin[2] ? B->X[2] : localBounds.Xmin[2];
    localBounds.Xmax[0] = B->X[0] > localBounds.Xmax[0] ? B->X[0] : localBounds.Xmax[0];
    localBounds.Xmax[1] = B->X[1] > localBounds.Xmax[1] ? B->X[1] : localBounds.Xmax[1];
    localBounds.Xmax[2] = B->X[2] > localBounds.Xmax[2] ? B->X[2] : localBounds.Xmax[2];
  }
  float localXmin[3], localXmax[3], globalXmin[3], globalXmax[3];
  for (int d=0; d<3; d++) {
    localXmin[d] = localBounds.Xmin[d];
    localXmax[d] = localBounds.Xmax[d];
  }
  MPI_Allreduce(localXmin, globalXmin, 3, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(localXmax, globalXmax, 3, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
  Bounds globalBounds;
  for (int d=0; d<3; d++) {
    real_t leeway = (globalXmax[d] - globalXmin[d]) * EPS;
    globalBounds.Xmin[d] = globalXmin[d] - leeway;
    globalBounds.Xmax[d] = globalXmax[d] + leeway;
  }
  for (int irank=0; irank<mpisize; irank++) {
    rankDispl[irank] = 0;
    rankCount[irank] = mpisize;
    rankColor[irank] = 0;
    rankKey[irank] = 0;
    rankMap[irank] = 0;
    sendDispl[irank] = 0;
    sendCount[irank] = numBodies;
    rankBounds[irank] = globalBounds;
  }
  for (int level=0; level<numLevels; level++) {
    int numPartitions = rankColor[mpisize-1] + 1;
    for (int ipart=0; ipart<numPartitions; ipart++) {
      int irank = rankMap[ipart];
      Bounds bounds = rankBounds[irank];
      int direction = 0, length = 0;
      for (int d=0; d<3; d++) {
	if (length < (bounds.Xmax[d] - bounds.Xmin[d])) {
	  direction = d;
	  length = (bounds.Xmax[d] - bounds.Xmin[d]);
	}
      }
      int numLocalBodies = sendCount[irank];
      int numGlobalBodies;
      MPI_Allreduce(&numLocalBodies, &numGlobalBodies, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
      int rankSplit = rankCount[irank] / 2;
      int oldRankCount = rankCount[irank];
      int bodySplit = numGlobalBodies * rankSplit / oldRankCount;
      int begin = 0;
      int end = sendCount[irank];
      int histOffset = 0;
      real_t xmax = bounds.Xmax[direction];
      real_t xmin = bounds.Xmin[direction];
      real_t dx = (xmax - xmin) / numBins;
      B_iter B0 = bodies.begin() + sendDispl[irank];
      if (bodySplit > 0) {
	for (int binLevel=0; binLevel<3; binLevel++) {
	  for (int ibin=0; ibin<numBins; ibin++) {
	    localHist[ibin] = 0;
	  }
	  for (int b=begin; b<end; b++) {
	    assert(sendDispl[irank]+b < numBodies);
	    real_t x = (B0+b)->X[direction];
	    int ibin = (x - xmin + EPS) / (dx + EPS);
	    assert(0 <= ibin);
	    assert(ibin < numBins);
	    localHist[ibin]++;
	  }
	  scanHist[0] = localHist[0];
	  for (int ibin=1; ibin<numBins; ibin++) {
	    scanHist[ibin] = scanHist[ibin-1] + localHist[ibin];
	  }
	  for (int b=end-1; b>=begin; b--) {
	    real_t x = (B0+b)->X[direction];
            int ibin = (x - xmin + EPS) / (dx + EPS);
	    scanHist[ibin]--;
	    int bnew = scanHist[ibin] + begin;
	    buffer[bnew] = *(B0+b);
	  }
	  for (int b=begin; b<end; b++) {
	    *(B0+b) = buffer[b];
	  }
	  MPI_Allreduce(localHist, globalHist, numBins, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	  int splitBin = 0;
	  while (histOffset < bodySplit) {
	    assert(splitBin < numBins);
	    histOffset += globalHist[splitBin];
	    splitBin++;
	  }
	  splitBin--;
	  histOffset -= globalHist[splitBin];
	  assert(0 <= splitBin);
	  assert(splitBin < numBins);
	  xmax = xmin + (splitBin + 1) * dx;
	  xmin = xmin + splitBin * dx;
	  dx = (xmax - xmin) / numBins;
	  scanHist[0] = 0;
          for (int ibin=1; ibin<numBins; ibin++) {
            scanHist[ibin] = scanHist[ibin-1] + localHist[ibin-1];
          }
	  begin += scanHist[splitBin];
	  end = begin + localHist[splitBin];
	}
      }
      int rankBegin = rankDispl[irank];
      int rankEnd = rankBegin + rankCount[irank];
      for (irank=rankBegin; irank<rankEnd; irank++) {
	rankSplit = rankCount[irank] / 2;
	if (irank - rankDispl[irank] < rankSplit) {
	  rankCount[irank] = rankSplit;
	  rankColor[irank] = rankColor[irank] * 2;
	  rankBounds[irank].Xmax[direction] = xmin;
	  sendCount[irank] = begin;
	} else {
	  rankDispl[irank] += rankSplit;
	  rankCount[irank] -= rankSplit;
	  rankColor[irank] = rankColor[irank] * 2 + 1;
	  rankBounds[irank].Xmin[direction] = xmin;
	  sendDispl[irank] += begin;
	  sendCount[irank] -= begin;
	}
	if (level == numLevels-1) rankColor[irank] = rankDispl[irank];
	rankKey[irank] = irank - rankDispl[irank];
	if (mpirank==0) printf("%d %d %d %d %d\n", irank, rankDispl[irank], rankCount[irank], sendDispl[irank], sendCount[irank]);
      }
    }
    int ipart = 0;
    for (int irank=0; irank<mpisize; irank++) {
      if (rankKey[irank] == 0) {
	rankMap[ipart] = rankDispl[irank];
	ipart++;
      }
    }
  }
  MPI_Finalize();
}
