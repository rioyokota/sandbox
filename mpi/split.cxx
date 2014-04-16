#include "mpi.h"
#include <stdio.h>

int main(int argc, char **argv) {
  int mpirank, mpisize;
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
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
  for (int irank=0; irank<mpisize; irank++) {
    rankDispl[irank] = 0;
    rankCount[irank] = mpisize;
    rankColor[irank] = 0;
    rankKey[irank] = 0;
  }
  if(mpirank==0) {
    for (int level=0; level<numLevels; level++) {
      for (int irank=0; irank<mpisize; irank++) {
	int rankSplit = rankCount[irank] / 2;
	if (irank - rankDispl[irank] < rankSplit) {
	  rankCount[irank] = rankSplit;
	  rankColor[irank] = rankColor[irank] * 2;
	} else {
	  rankCount[irank] -= rankSplit;
	  rankDispl[irank] += rankSplit;
	  rankColor[irank] = rankColor[irank] * 2 + 1;
	}
	if (level == numLevels-1) rankColor[irank] = rankDispl[irank];
	rankKey[irank] = irank - rankDispl[irank];
      }
      printf("\nLevel %d Displ:",level);
      for (int irank=0; irank<mpisize; irank++) {
	printf(" %2d",rankDispl[irank]);
      }
      printf("\nLevel %d Count:",level);
      for (int irank=0; irank<mpisize; irank++) {
	printf(" %2d",rankCount[irank]);
      }
      printf("\nLevel %d Color:",level);
      for (int irank=0; irank<mpisize; irank++) {
	printf(" %2d",rankColor[irank]);
      }
      printf("\nLevel %d Key  :",level);
      for (int irank=0; irank<mpisize; irank++) {
	printf(" %2d",rankKey[irank]);
      }
      printf("\n");
    }
  }
  MPI_Finalize();
}
