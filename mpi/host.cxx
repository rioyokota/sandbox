#include <mpi.h>
#include <iostream>
#include <unistd.h>

int main(int argc, char **argv) {
  char hostname[256];                                           // Define hostname
  int size,rank,len;                                            // Define MPI size and rank
  MPI_Init(&argc,&argv);                                        // Initialize MPI communicator
  MPI_Comm_size(MPI_COMM_WORLD,&size);                          // Get number of MPI processes
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);                          // Get rank of current MPI process
  MPI_Get_processor_name(hostname,&len);                        // Get hostname
  for( int irank=0; irank!=size; ++irank ) {                    // Loop over MPI ranks
    MPI_Barrier(MPI_COMM_WORLD);                                //  Synchronize processes
    if( rank == irank ) {                                       //  If loop counter matches MPI rank
      std::cout << hostname << " " << rank << " / " << size << std::endl;// Print hostname, rank, and size
    }                                                           //  Endif for loop counter
    usleep(100);                                                //  Wait for 100 microseconds
  }                                                             // End loop over MPI ranks
  MPI_Finalize();                                               // Finalize MPI communicator
}
