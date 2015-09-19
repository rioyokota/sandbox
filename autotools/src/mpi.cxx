#include <mpi.h>
#include <iostream>
int main(int argc, char ** argv) {
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::cout << "myrank: " << rank << std::endl;
  MPI_Finalize();
}
