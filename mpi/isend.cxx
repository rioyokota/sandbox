#include <mpi.h>
#include <iostream>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main(int argc, char **argv) {
  const int N = 1000000;
  int size,rank;
  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD,&size);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  float * send = new float [N];
  float * recv = new float [N];
  int sendRank = (rank + 1) % size;
  int recvRank = (rank - 1 + size) % size;
  MPI_Request sendReq, recvReq;
  double tic = get_time();
  MPI_Isend(send, N, MPI_FLOAT, recvRank, 0, MPI_COMM_WORLD, &sendReq);
  MPI_Irecv(recv, N, MPI_FLOAT, sendRank, 0, MPI_COMM_WORLD, &recvReq);
  double toc = get_time();
  if (rank == 0) std::cout << "Send : " << toc-tic << std::endl;
  tic = get_time();
  MPI_Wait(&sendReq, MPI_STATUS_IGNORE);
  MPI_Wait(&recvReq, MPI_STATUS_IGNORE);
  toc = get_time();
  if (rank == 0) std::cout << "Recv : " << toc-tic << std::endl;
  delete[] send;
  delete[] recv;
  MPI_Finalize();
}
