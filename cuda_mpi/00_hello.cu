#include <mpi.h>
#include <cstdio>

__global__ void mykernel(void) {
  printf("Hello GPU\n");
}

int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  int mpisize, mpirank;
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  mykernel<<<1,1>>>();
  cudaDeviceSynchronize();
  printf("rank: %d/%d\n",mpirank,mpisize);
  MPI_Finalize();
}
