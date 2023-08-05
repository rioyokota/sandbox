#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <random>
#include <sys/time.h>
#include <omp.h>

namespace sc {
  constexpr int maxThreads = 48;
  constexpr int ranking = 1000;
  double TIME0;
  double *X, *Y;

  struct Pair {
    int i, j;
    double dist2;

    bool operator<(const Pair &rhs) const {
      return dist2 < rhs.dist2;
    }
  };
  Pair *pairs;

  double get_time() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (double)(tv.tv_sec+tv.tv_usec*1e-6);
  }

  void input(int N) {
    X = new double [N];
    Y = new double [N];
    pairs = new Pair [ranking];
    std::uniform_real_distribution<double> dis(0.0, 1.0);
#pragma omp parallel for
    for (int ib=0; ib<maxThreads; ib++) {
      std::mt19937 generator(ib);
      int begin = ib * (N / maxThreads);
      int end = (ib + 1) * (N / maxThreads);
      if(ib == maxThreads-1) end = N > end ? N : end;
      for (int i=begin; i<end; i++) {
        X[i] = dis(generator);
        Y[i] = dis(generator);
      }
    }
    TIME0 = get_time();
  }

  void output() {
    printf("%e s\n",get_time()-TIME0);
    for(size_t k : {0, 9, 99, 999}) {
      const Pair &pp = pairs[k];
      printf("#%4zu : %10.15e (%d,%d)\n", k+1, sqrt(pp.dist2), pp.i, pp.j);
    }
  }

  void finalize() {
    delete[] X;
    delete[] Y;
    delete[] pairs;
  }
};
