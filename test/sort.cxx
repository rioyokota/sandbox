#include <assert.h>
#include <algorithm>
#include <iostream>
#include <sys/time.h>
#include <vector>

typedef std::vector<int> Index;
typedef std::vector<int>::iterator I_iter;
typedef std::vector<int>::reverse_iterator I_rter;

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return double(tv.tv_sec+tv.tv_usec*1e-6);
}

int main() {
  double tic, toc;
  tic = get_time();
  Index index(10000000);
  toc = get_time();
  std::cout << "init : " << toc-tic << std::endl;

  tic = get_time();
  for( int i=0; i!=int(index.size()); ++i ) index[i] = i / 10;
  toc = get_time();
  std::cout << "index: " << toc-tic << std::endl;

  tic = get_time();
  std::random_shuffle(index.begin(),index.end());
  toc = get_time();
  std::cout << "rand : " << toc-tic << std::endl;

  Index buffer = index;
  tic = get_time();
#if 0
  std::sort(index.begin(),index.end());
#else
  Index bucket(1000000);
  for( I_iter I=buffer.begin(); I!=buffer.end(); ++I ) bucket[*I]++;
  for( I_iter I=bucket.begin()+1; I!=bucket.end(); ++I ) *I += *(I-1);
  for( I_rter I=buffer.rbegin(); I!=buffer.rend(); ++I ) index[--bucket[*I]] = *I;
#endif
  toc = get_time();
  std::cout << "sort : " << toc-tic << std::endl;
  for( int i=1; i!=int(index.size()); ++i ) assert( index[i] >= index[i-1] );

  tic = get_time();
  std::nth_element(index.begin(),index.begin()+index.size()/2,index.end());
  toc = get_time();
  std::cout << "nth  : " << toc-tic << std::endl;
}
