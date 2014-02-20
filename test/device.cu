#include <mpi.h>
#include <stdio.h>

__global__ void GPU_Kernel() {
  printf(" GPU thread : %d / %d  GPU block  : %d / %d\n",
         threadIdx.x, blockDim.x, blockIdx.x, gridDim.x);       // Print GPU thread & block rank / size
}

int main(int argc, char **argv) {
  int mpisize, mpirank, gpusize, gpurank;                       // Define variables
  cudaGetDeviceCount(&gpusize);                                 // Get number of GPUs
  MPI_Init(&argc, &argv);                                       // Initialize MPI communicator
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);                      // Get number of MPI processes
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);                      // Get rank of current MPI process
  cudaSetDevice(mpirank % gpusize);                             // Set GPU device ID
  cudaGetDevice(&gpurank);                                      // Get GPU device ID
  for (int irank=0; irank!=mpisize; irank++) {                  // Loop over MPI ranks
    MPI_Barrier(MPI_COMM_WORLD);                                //  Synchronize processes
    if (mpirank == irank) {                                     //  If loop counter matches MPI rank
      printf("MPI rank    : %d / %d  GPU device : %d / %d\n",
             mpirank, mpisize, gpurank, gpusize);               //   Print MPI & GPU rank / size
      GPU_Kernel<<<2,2>>>();                                    //   Launch GPU kernel
      cudaThreadSynchronize();                                  //   Flush GPU printf buffer
    }                                                           //  Endif for loop counter
  }                                                             // End loop over MPI ranks
  MPI_Finalize();                                               // Finalize MPI communicator
}
