#include <cublas_v2.h>
#include <magma_v2.h>

int main(int argc, char **argv) {
  int m = 128;
  int n = 64;
  int k = 32;
  cublasStatus_t cublasStat = cublasCreate(&handle);
  cublasStat = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
  size_t matrixSizeA = (size_t)m * k;
  size_t matrixSizeB = (size_t)k * n;
  size_t matrixSizeC = (size_t)m * n;
  double **devPtrA = 0, **devPtrB = 0, **devPtrC = 0;
  cudaMalloc((void**)&devPtrA[0], matrixSizeA * sizeof(devPtrA[0][0]));
  cudaMalloc((void**)&devPtrB[0], matrixSizeB * sizeof(devPtrB[0][0]));
  cudaMalloc((void**)&devPtrC[0], matrixSizeC * sizeof(devPtrC[0][0]));
  double A  = (double *)malloc(matrixSizeA * sizeof(A[0]));
  double B  = (double *)malloc(matrixSizeB * sizeof(B[0]));
  double C  = (double *)malloc(matrixSizeC * sizeof(C[0]));
  memset( A, 0xFF, matrixSizeA* sizeof(A[0]));
  memset( B, 0xFF, matrixSizeB* sizeof(B[0]));
  memset( C, 0xFF, matrixSizeC* sizeof(C[0]));
  cublasSetMatrix(m, k, sizeof(A[0]), A, rowsA, devPtrA[i], rowsA);
  cublasSetMatrix(k, n, sizeof(B[0]), B, rowsB, devPtrB[i], rowsB);
  cublasSetMatrix(m, n, sizeof(C[0]), C, rowsC, devPtrC[i], rowsC);
  double alpha = 1;
  double beta = 1;
  cublasStat = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, alpha,
                            A, CUDA_R_16F, m, B, CUDA_R_16F, k,
                            beta, C, CUDA_R_16F, m, CUDA_R_32F, algo);
  return 0;
}