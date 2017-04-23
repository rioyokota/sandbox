#include "mpi.h"
#include <cstdio>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  int data[4] = {0,0,0,0};
  if(!mpirank) {
    for(int i=0; i<4; i++) data[i] = i+1;
  }
  printf("rank%d: before [%d %d %d %d]\n",mpirank,data[0],data[1],data[2],data[3]);
  MPI_Bcast(data, 4, MPI_INT, 0, MPI_COMM_WORLD);
  printf("rank%d: after  [%d %d %d %d]\n",mpirank,data[0],data[1],data[2],data[3]);
  MPI_Finalize();
}
