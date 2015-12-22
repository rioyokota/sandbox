/*******************************************************************************
*  Jeremiah Palmer
*  jeremiahpalmer@gmail.com
 
*  Purpose:  This program tests the performance of various parallelizations
*  of dgemm.  Currently, it will test:
*  (1)  A single multi-threaded MKL dgemm using a range of threads.
*  (2)  Several MKL dgemms called within a parallel region.
*  (3)  Several MKL dgemms called within a nested parallel region. 
*
*  Each test provides the flop rate, in Gigaflops per second for each parallel
*  scheme.  These tests demonstrate the failing of resources when the number
*  of threads nears the number of CPU cores.
*
*******************************************************************************/

#ifndef MAIN
#define MAIN

#include <stdio.h>
#include <omp.h>
#include <mkl.h>
#include <iostream>
#include <string>
#include <sys/time.h>

//--- namespace ----------------------------------------------------------------
//using namespace std;

double compute_speed(MKL_INT n, double time)
{
  double Tn, gflops;
  Tn = ((double) n)/1000.0;
  gflops = 2.0*Tn*Tn*Tn/time;
  return(gflops);
} 

void fill_matrix(MKL_INT n, double *A){
  unsigned short seed[3];
  seed[0] = 1;
  seed[1] = 1;
  seed[2] = 1;
  seed48(seed);

  for (MKL_INT j=0; j<n*n; j++){
    *(A + j) = drand48();
  }
}

double get_time(){
  struct timeval tp;
  struct timezone tzp;
  MKL_INT i = gettimeofday(&tp,&tzp);
  return((double) tp.tv_sec + (double) tp.tv_usec*1.e-6);
}


//--- main ---------------------------------------------------------------------
int main(int argc, char **argv)
{
  MKL_INT n, p;
  MKL_INT size=1000;
  double alpha=1.2, beta = 2.3;
  double time;
  n = size;
  if (argc > 1) {
    n = atoi(argv[1]);
  }
    
  p = omp_get_max_threads();

  printf("\nSize = %i\n", n);

  //  (1)  Call a single multi-threaded MKL DGEMM.  Note that although the
  //       compute speed varies greatly with higher number of threads, the max
  //       number of threads always gives the best performance.
  printf("A single multi-threaded MKL DGEMM\n");  
  double *A1, *B1, *C1;
  A1 = (double *)malloc(n*n*sizeof(double));
  B1 = (double *)malloc(n*n*sizeof(double));
  C1 = (double *)malloc(n*n*sizeof(double));
  fill_matrix(n, A1);
  fill_matrix(n, B1);
  fill_matrix(n, C1);

  for (MKL_INT nthreads=1; nthreads<=p; nthreads++){
    omp_set_num_threads(nthreads);
    mkl_set_num_threads(nthreads);

    time = get_time(); 
    dgemm("N", "N", &n, &n, &n, &alpha, A1, &n, B1, &n, &beta, C1, &n);
    time = get_time() - time;

    printf("#threads = %i  Compute Speed = %.2f GFLOP/sec\n", nthreads, 
           compute_speed(n, time));
  }

  free(A1); 
  free(B1); 
  free(C1); 

  //  (2)  Call multiple serial MKL DGEMMs in parallel.  Note that the compute
  //       speed varies greatly for all higher numbers of threads.  There 
  //       is no benefit to using higher numbers of threads.
  printf("Multiple MKL DGEMMs called in parallel\n"); 

  for (MKL_INT nthreads=1; nthreads <= p; nthreads++){ 
    time = -1.0;
    omp_set_num_threads(nthreads);
    mkl_set_dynamic(false);
    #pragma omp parallel reduction(max: time)
    {
      mkl_set_num_threads(1);
      double *A, *B, *C;
      A = (double *)malloc(n*n*sizeof(double));
      B = (double *)malloc(n*n*sizeof(double));
      C = (double *)malloc(n*n*sizeof(double));
      fill_matrix(n, A);
      fill_matrix(n, B);
      fill_matrix(n, C);

      time = get_time();
      dgemm("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
      time = get_time() - time;
 
      free(A);
      free(B);
      free(C);
    }

    printf("#threads = %i  Total Compute Speed = %.2f GFLOP/sec\n", nthreads, 
           nthreads*compute_speed(n, time));
  }

  //  (3)  Call multiple 2-threaded MKL DGEMMs in a nested parallel region.
  printf("Multiple MKL DGEMMs called in nested parallel\n"); 

  for (MKL_INT nthreads=1; nthreads <= p/2; nthreads++){ 
    time = -1.0;
    omp_set_num_threads(nthreads);
    omp_set_nested(true);
    mkl_set_dynamic(false);
    #pragma omp parallel reduction(max: time)
    {
      mkl_set_num_threads(2);
      double *A, *B, *C;
      A = (double *)malloc(n*n*sizeof(double));
      B = (double *)malloc(n*n*sizeof(double));
      C = (double *)malloc(n*n*sizeof(double));
      fill_matrix(n, A);
      fill_matrix(n, B);
      fill_matrix(n, C);

      time = get_time();
      dgemm("N", "N", &n, &n, &n, &alpha, A, &n, B, &n, &beta, C, &n);
      time = get_time() - time;
 
      free(A);
      free(B);
      free(C);
    }

    printf("#threads = %i  Total Compute Speed = %.2f GFLOP/sec\n", nthreads*2,
           nthreads*compute_speed(n, time));
  }
  return 0; 
}

//------------------------------------------------------------------------------
#endif


