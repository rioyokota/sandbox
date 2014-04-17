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
  int sendCount[mpisize];
  Bodies bodies(numBodies);
  for (B_iter B=bodies.begin(); B!=bodies.end(); B++) {
    B->X[0] = drand48();
    B->X[1] = drand48();
    B->X[2] = drand48();
  }
  for (int irank=0; irank<mpisize; irank++) {
    rankDispl[irank] = 0;
    rankCount[irank] = mpisize;
    rankColor[irank] = 0;
    rankKey[irank] = 0;
    rankMap[irank] = 0;
    sendCount[irank] = numBodies;
  }
  if(mpirank==0) {
    for (int level=0; level<numLevels; level++) {
      int numPartitions = rankColor[mpisize-1] + 1;
      int total = 0;
      for (int ipart=0; ipart<numPartitions; ipart++) {
	total += sendCount[rankMap[ipart]];
      }
      printf("%d %2d %d\n",level,numPartitions,total);
      int ipart = 0;
      for (int irank=0; irank<mpisize; irank++) {
	int rankSplit = rankCount[irank] / 2;
	int oldRankCount = rankCount[irank];
	if (irank - rankDispl[irank] < rankSplit) {
	  rankCount[irank] = rankSplit;
	  rankColor[irank] = rankColor[irank] * 2;
	  sendCount[irank] = (sendCount[irank] * rankSplit) / oldRankCount;
	} else {
	  rankCount[irank] -= rankSplit;
	  rankDispl[irank] += rankSplit;
	  rankColor[irank] = rankColor[irank] * 2 + 1;
	  sendCount[irank] -= (sendCount[irank] * rankSplit) / oldRankCount;
	}
	if (level == numLevels-1) rankColor[irank] = rankDispl[irank];
	rankKey[irank] = irank - rankDispl[irank];
	if (rankKey[irank] == 0) {
	  rankMap[ipart] = rankDispl[irank];
	  ipart++;
	}
      }
    }
    int numPartitions = rankColor[mpisize-1] + 1;
    int total = 0;
    for (int ipart=0; ipart<numPartitions; ipart++) {
      total += sendCount[rankMap[ipart]];
    }
  }
  MPI_Finalize();
}
