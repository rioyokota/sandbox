#include "mpi.h"
#include <stdio.h>

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
  int rankOther[mpisize];
  int rankColor[mpisize];
  int rankKey[mpisize];
  int rankMap[mpisize];
  int sendCount[mpisize];
  for (int irank=0; irank<mpisize; irank++) {
    rankDispl[irank] = 0;
    rankCount[irank] = mpisize;
    rankOther[irank] = mpisize;
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
      int ipart = 0;
      for (int irank=0; irank<mpisize; irank++) {
	int rankSplit = rankCount[irank] / 2;
	if (irank - rankDispl[irank] < rankSplit) {
	  rankOther[irank] = rankCount[irank] - rankSplit;
	  rankCount[irank] = rankSplit;
	  rankColor[irank] = rankColor[irank] * 2;
	  sendCount[irank] = (sendCount[irank] * rankCount[irank]) / (rankCount[irank] + rankOther[irank]);
	} else {
	  rankOther[irank] = rankSplit;
	  rankCount[irank] -= rankSplit;
	  rankDispl[irank] += rankSplit;
	  rankColor[irank] = rankColor[irank] * 2 + 1;
	  sendCount[irank] -= (sendCount[irank] * rankOther[irank]) / (rankCount[irank] + rankOther[irank]);
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
