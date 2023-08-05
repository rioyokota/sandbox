#include <sc_header.hpp>
#include <queue>

int main(int argc, char **argv){
  int N = 1024;
  if(argc > 1)  N = atoi(argv[1]);
  sc::input(N);

  std::priority_queue<sc::Pair> queue;
  double r2_max = 2.0;
  for(int i=0; i<N; i++){
    for(int j=i+1; j<N; j++){
      double dx = sc::X[j] - sc::X[i];
      double dy = sc::Y[j] - sc::Y[i];
      double r2 = dx*dx;
      r2 += dy * dy;
      if(r2 < r2_max){
	if(queue.size() < 1000){
	  queue.push({i, j, r2});
	}else if (r2 < queue.top().dist2) {
	  queue.pop();
	  queue.push({i, j, r2});
	  r2_max = queue.top().dist2;
	}
      }
    }
  }
  for(int i=0; i<1000; i++){
    sc::pairs[999-i] = queue.top();
    queue.pop();
  }
  sc::output();
  sc::finalize();
  return 0;
};
