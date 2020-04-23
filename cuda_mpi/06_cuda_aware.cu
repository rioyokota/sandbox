#include <mpi.h>
#include <cstdio>
#include <sys/time.h>
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec)+double(tv.tv_usec)*1e-6;
}
__global__ void GPU_Kernel(int *send) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  send[i] = 1;
}

#define N (2048*2048)
#define M 512

int main(int argc, char **argv) {
  int mpisize, mpirank;
  int size = N * sizeof(int);
  int *send = (int *)malloc(size);
  int *recv = (int *)malloc(size);
  int *d_send, *d_recv;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  cudaSetDevice(mpirank % mpisize);
  cudaMalloc((void **) &d_send, size);
  cudaMalloc((void **) &d_recv, size);
  double tic = get_time();
  cudaMemcpy(d_send, send, size, cudaMemcpyHostToDevice);
  double toc = get_time();
  if(mpirank==0) printf("%-25s : %lf s\n", "Memcpy host to device", toc-tic);
  GPU_Kernel<<<N/M,M>>>(d_send);
  cudaThreadSynchronize();
  tic = get_time();
  if(mpirank==0) printf("%-25s : %lf s\n", "CUDA kernel", tic-toc);
  int sendrank = (mpirank + 1) % mpisize;
  int recvrank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Request reqs[2];
  MPI_Status stats[2];
  MPI_Isend(d_send, N, MPI_INT, sendrank, 0, MPI_COMM_WORLD, &reqs[0]);
  MPI_Irecv(d_recv, N, MPI_INT, recvrank, 0, MPI_COMM_WORLD, &reqs[1]);
  MPI_Waitall(2, reqs, stats);
  toc = get_time();
  if(mpirank==0) printf("%-25s : %lf s\n", "MPI communication", toc-tic);
  cudaMemcpy(recv, d_recv, size, cudaMemcpyDeviceToHost);
  tic = get_time();
  if(mpirank==0) printf("%-25s : %lf s\n", "Memcpy device to host", tic-toc);
  int sum = 0;
  for (int i=0; i<N; i++)
    sum += recv[i];
  for (int irank=0; irank<mpisize; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpirank == irank) {
      printf("rank%d: sum=%d, N=%d\n", mpirank, sum, N);
    }
  }
  free(send); free(recv);
  cudaFree(d_send); cudaFree(d_recv);
  MPI_Finalize();
}