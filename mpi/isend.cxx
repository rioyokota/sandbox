#include <mpi.h>
#include <cmath>
#include <iostream>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char **argv) {
  const int N = 10000000;
  int size, rank;
  double tic, toc;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  float * send = new float [N];
  float * recv = new float [N];
  for (int i=0; i<N; i++) send[i] = rank;
  int sendRank = (rank + 1) % size;
  int recvRank = (rank - 1 + size) % size;
  MPI_Request sendReq, recvReq;
  double TIC = get_time();
#pragma omp parallel sections private(tic, toc)
  {
#pragma omp section
    {
      tic = get_time();
      MPI_Isend(send, N, MPI_FLOAT, recvRank, 0, MPI_COMM_WORLD, &sendReq);
      MPI_Irecv(recv, N, MPI_FLOAT, sendRank, 0, MPI_COMM_WORLD, &recvReq);
      MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
      MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
      toc = get_time();
      if (rank == 0) std::cout << "Send : " << toc-tic << std::endl;
    }
#pragma omp section
    {
      tic = get_time();
      for (int i=0; i<N; i++) {
	recv[i] = exp(i * M_PI);
      }
      toc = get_time();
      if (rank == 0) std::cout << "Calc : " << toc-tic << std::endl; 
    }
  }
  double TOC = get_time();
  if (rank == 0) std::cout << "Total: " << TOC-TIC << std::endl;
  float sum = 0;
  for (int i=0; i<N; i++) sum += recv[i];
  //std::cout << "Rank : " << rank << " Sum : " << sum << std::endl;
  delete[] send;
  delete[] recv;
  MPI_Finalize();
}
