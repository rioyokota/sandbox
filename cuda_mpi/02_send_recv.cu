#include <mpi.h>
#include <cstdio>
__global__ void GPU_Kernel(int *send) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  send[i] += 10 * i;
}
int main(int argc, char **argv) {
  int mpisize, mpirank;
  int size = 4 * sizeof(int);
  int *send = (int *)malloc(size);
  int *recv = (int *)malloc(size);
  int *d_send, *d_recv;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &mpisize);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
  for(int i=0; i<4; i++)
    send[i] = mpirank;
  cudaSetDevice(mpirank % mpisize);
  cudaMalloc((void **) &d_send, size);
  cudaMalloc((void **) &d_recv, size);
  cudaMemcpy(d_send, send, size, cudaMemcpyHostToDevice);
  GPU_Kernel<<<2,2>>>(d_send);
  cudaMemcpy(send, d_send, size, cudaMemcpyDeviceToHost);
  int sendrank = (mpirank + 1) % mpisize;
  int recvrank = (mpirank - 1 + mpisize) % mpisize;
  MPI_Send(send, 4, MPI_INT, sendrank, 0, MPI_COMM_WORLD);
  MPI_Recv(recv, 4, MPI_INT, recvrank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  for (int irank=0; irank<mpisize; irank++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (mpirank == irank) {
      printf("rank%d: send_rank=%d, recv_rank=%d\n", mpirank, sendrank, recvrank);
      printf("send=[%d %d %d %d], recv=[%d %d %d %d]\n",
             send[0],send[1],send[2],send[3],recv[0],recv[1],recv[2],recv[3]);
    }
  }
  free(send); free(recv);
  cudaFree(d_send); cudaFree(d_recv);
  MPI_Finalize();
}